from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from utils.prompts import llm_judge_prompt
from database.evaluation import EvaluationStorage
import logging
import os
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class JudgeOutput:
    relevance: int  # 0-5 scale
    accuracy: int  # 0-5 scale
    pii_violation: int  # 0 or 1
    safety_violation: int  # 0 or 1
    clarity: int  # 0-5 scale
    overall_score: int  # 0-5 scale
    rationale: str

class EvaluationService(ABC):
    def __init__(self, db_path: str = "./db/evaluation.db"):
        self.db_path = db_path
        self.storage = EvaluationStorage(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def evaluate(self, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> JudgeOutput:
        """
        Abstract method to evaluate a chatbot response.
        Must be implemented by all subclasses.

        Args:
            user_query: The user's input query
            agent_response: The chatbot's response
            conversation_id: Optional conversation ID for linking
            message_id: Optional message ID for linking

        Returns:
            JudgeOutput: Structured evaluation result
        """
        pass

    def save_evaluation(self, evaluation: JudgeOutput, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> int:
        """Save evaluation result to the database"""
        return self.storage.add_evaluation(
            user_query=user_query,
            assistant_response=agent_response,
            relevance=evaluation.relevance,
            accuracy=evaluation.accuracy,
            pii_violation=evaluation.pii_violation,
            safety_violation=evaluation.safety_violation,
            clarity=evaluation.clarity,
            overall_score=evaluation.overall_score,
            rationale=evaluation.rationale,
            conversation_id=conversation_id,
            message_id=message_id
        )

class LLMJudge(EvaluationService):
    def __init__(self, model_name: str = "gpt-5-chat", db_path: str = "./db/evaluation.db"):
        super().__init__(db_path)
        self.model = AzureChatOpenAI(
            azure_deployment=model_name,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
        )
        self.judge_agent = create_agent(
            model=self.model,
            tools=[],
            response_format=JudgeOutput,
        )

    async def evaluate(self, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> JudgeOutput:
        try:
            # Format the judge prompt with the query and response
            formatted_prompt = llm_judge_prompt.format(
                user_query=user_query,
                assistant_response=agent_response
            )
            messages = [SystemMessage(content=llm_judge_prompt)]
            messages.append(HumanMessage(content=formatted_prompt))

            # Invoke the judge agent
            response = await self.judge_agent.ainvoke({"messages": messages})
            judge_output = response.get('structured_response')

            logger.info(f"LLM Judge response: {judge_output}")

            if judge_output is None:
                self.logger.error("LLM Judge returned no response")
                judge_output = self._get_failed_evaluation("No response from judge model")
            else:
                # Ensure pii_violation and safety_violation have default values if None
                if judge_output.pii_violation is None:
                    judge_output.pii_violation = 0
                if judge_output.safety_violation is None:
                    judge_output.safety_violation = 0

            # Save to database
            self.save_evaluation(judge_output, user_query, agent_response, conversation_id, message_id)
            logger.info("Evaluation saved to database")
            logger.info(f"Evaluation details: {judge_output}")
            return judge_output

        except Exception as e:
            self.logger.error(f"LLM Judge evaluation failed: {str(e)}")
            return self._get_failed_evaluation(f"Evaluation error: {str(e)}")

    def _get_failed_evaluation(self, reason: str) -> JudgeOutput:
        """Return a default failed evaluation output"""
        return JudgeOutput(
            relevance=0,
            accuracy=0,
            pii_violation=0,
            safety_violation=0,
            clarity=0,
            overall_score=0,
            rationale=f"Evaluation failed - {reason}"
        )

class LightHeuristic:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_heuristics( self, agent_response: str, retrieved_docs: Optional[List[Dict[str, Any]]] = None, response_time_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate lightweight heuristics for a chatbot response.

        Args:
            agent_response: The chatbot's response text
            retrieved_docs: Optional list of retrieved documents with metadata
            response_time_ms: Optional response time in milliseconds

        Returns:
            Dictionary containing heuristic metrics
        """
        try:
            heuristics = {
                "response_length": self._calculate_response_length(agent_response),
                "has_citation": self._check_citations(agent_response),
                "retrieval_confidence": self._calculate_retrieval_confidence(retrieved_docs),
                "response_time_ms": response_time_ms if response_time_ms is not None else 0,
                "num_docs_retrieved": len(retrieved_docs) if retrieved_docs else 0
            }

            self.logger.info(f"Calculated heuristics: {heuristics}")
            return heuristics

        except Exception as e:
            self.logger.error(f"Error calculating heuristics: {str(e)}")
            return self._get_default_heuristics()

    def _calculate_response_length(self, response: str) -> int:
        """Calculate the length of the response in characters."""
        return len(response.strip())

    def _check_citations(self, response: str) -> bool:
        """
        Check if the response contains citations or references.
        Looks for common citation patterns like [1], (Source), etc.
        """
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d+\)',  # (1), (2), etc.
            r'\[.*?\]',  # [Source], [Document], etc.
            r'(?i)(source:|reference:|according to)',  # Text-based citations
        ]

        for pattern in citation_patterns:
            if re.search(pattern, response):
                return True
        return False

    def _calculate_retrieval_confidence(self,retrieved_docs: Optional[List[Dict[str, Any]]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        Uses similarity scores or relevance scores if available.
        """
        if not retrieved_docs:
            return 0.0

        try:
            # Try to extract scores from retrieved documents
            scores = []
            for doc in retrieved_docs:
                # Check for common score field names
                score = (
                    doc.get('score') or
                    doc.get('similarity') or
                    doc.get('relevance_score') or
                    doc.get('confidence')
                )
                if score is not None:
                    scores.append(float(score))

            if scores:
                # Return average score
                return round(sum(scores) / len(scores), 2)
            else:
                # If no scores available, return a default based on number of docs
                # More docs retrieved suggests higher confidence up to a point
                return min(0.5 + (len(retrieved_docs) * 0.1), 1.0)

        except Exception as e:
            self.logger.warning(f"Error calculating retrieval confidence: {str(e)}")
            return 0.0

    def _get_default_heuristics(self) -> Dict[str, Any]:
        """Return default heuristics when calculation fails."""
        return {
            "response_length": 0,
            "has_citation": False,
            "retrieval_confidence": 0.0,
            "response_time_ms": 0,
            "num_docs_retrieved": 0
        }

@dataclass
class SemanticEvaluationResult:
    """Result of semantic similarity evaluation"""
    query_response_similarity: float
    response_source_max_similarity: float
    response_source_avg_similarity: float
    query_response_keyword_match: Dict[str, Any]
    source_response_keyword_match: Dict[str, Any]
    overall_semantic_score: float
    details: Dict[str, Any]

class SemanticSimilarityEvaluator:
    """
    Evaluates agent responses using semantic similarity and keyword matching.

    Metrics:
    1. Query-Response Similarity: Cosine similarity between query and response embeddings
    2. Response-Source Document Similarity: Max/avg similarity between response and source docs
    3. Keyword Extraction and Matching: Q-R and D-R keyword overlap analysis
    """

    def __init__(self, embedding_model : HuggingFaceEmbeddings , spacy_model: str = "en_core_web_sm"):
        """
        Initialize the evaluator with embedding and NLP models.

        Args:
            embedding_model: Pre-initialized SentenceTransformer model for generating embeddings
            spacy_model: SpaCy model to use for keyword extraction (default: en_core_web_sm)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use the provided embedding model
        self.embedding_model = embedding_model
        self.logger.info(f"Using provided embedding model: {type(embedding_model).__name__}")

        # Load spaCy model for keyword extraction
        try:
            self.nlp = spacy.load(spacy_model)
            self.logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            self.logger.warning(f"SpaCy model '{spacy_model}' not found. Attempting to download...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)

    def evaluate( self, user_query: str, agent_response: str, source_documents: Optional[List[str]] = None) -> SemanticEvaluationResult:
        """
        Perform comprehensive semantic evaluation of agent response.

        Args:
            user_query: The user's input query
            agent_response: The agent's response
            source_documents: List of source document texts used for the response

        Returns:
            SemanticEvaluationResult with all evaluation metrics
        """
        try:
            self.logger.info("Starting semantic evaluation...")

            # 1. Query-Response Similarity
            qr_similarity = self._compute_query_response_similarity(user_query, agent_response)

            # 2. Response-Source Document Similarity
            if source_documents:
                rs_max_sim, rs_avg_sim, doc_similarities = self._compute_response_source_similarity(
                    agent_response, source_documents
                )
            else:
                rs_max_sim, rs_avg_sim, doc_similarities = 0.0, 0.0, []

            # 3. Keyword Extraction and Matching
            qr_keyword_match = self._compute_query_response_keyword_match(user_query, agent_response)

            # 4. Source-Response Keyword Matching
            if source_documents:
                sr_keyword_match = self._compute_source_response_keyword_match(
                    source_documents, agent_response
                )
            else:
                sr_keyword_match = {
                    "match_ratio": 0.0,
                    "matched_keywords": [],
                    "source_keywords": [],
                    "response_keywords": []
                }

            # 5. Calculate overall semantic score
            overall_score = self._calculate_overall_score(
                qr_similarity,
                rs_max_sim if source_documents else qr_similarity,  # Use qr_sim if no sources
                qr_keyword_match["match_ratio"],
                sr_keyword_match["match_ratio"]
            )

            result = SemanticEvaluationResult(
                query_response_similarity=qr_similarity,
                response_source_max_similarity=rs_max_sim,
                response_source_avg_similarity=rs_avg_sim,
                query_response_keyword_match=qr_keyword_match,
                source_response_keyword_match=sr_keyword_match,
                overall_semantic_score=overall_score,
                details={
                    "document_similarities": doc_similarities,
                    "num_source_documents": len(source_documents) if source_documents else 0
                }
            )

            self.logger.info(f"Semantic evaluation complete. Overall score: {overall_score:.3f}")
            return result

        except Exception as e:
            self.logger.error(f"Semantic evaluation failed: {str(e)}")
            raise

    def _compute_query_response_similarity(self, query: str, response: str) -> float:
        """
        Compute cosine similarity between query and response embeddings.

        Args:
            query: User query text
            response: Agent response text

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # HuggingFaceEmbeddings uses embed_query for single text
            query_embedding = np.array([self.embedding_model.embed_query(query)])
            response_embedding = np.array([self.embedding_model.embed_query(response)])

            similarity = cosine_similarity(query_embedding, response_embedding)[0][0]

            self.logger.info(f"Query-Response similarity: {similarity:.3f}")
            return float(similarity)

        except Exception as e:
            self.logger.error(f"Error computing Q-R similarity: {str(e)}")
            return 0.0

    def _compute_response_source_similarity( self, response: str, source_documents: List[str]) -> tuple[float, float, List[float]]:
        """
        Compute cosine similarity between response and each source document.

        Args:
            response: Agent response text
            source_documents: List of source document texts

        Returns:
            Tuple of (max_similarity, avg_similarity, list_of_similarities)
        """
        try:
            if not source_documents:
                return 0.0, 0.0, []

            # HuggingFaceEmbeddings uses embed_query for single text and embed_documents for lists
            response_embedding = np.array([self.embedding_model.embed_query(response)])
            doc_embeddings = np.array(self.embedding_model.embed_documents(source_documents))

            similarities = []
            for i, doc_embedding in enumerate(doc_embeddings):
                sim = cosine_similarity(response_embedding, [doc_embedding])[0][0]
                similarities.append(float(sim))
                self.logger.debug(f"Response-Doc[{i}] similarity: {sim:.3f}")

            max_sim = max(similarities) if similarities else 0.0
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

            self.logger.info(f"Response-Source similarity - Max: {max_sim:.3f}, Avg: {avg_sim:.3f}")
            return max_sim, avg_sim, similarities

        except Exception as e:
            self.logger.error(f"Error computing R-S similarity: {str(e)}")
            return 0.0, 0.0, []

    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract important keywords from text using spaCy NLP.

        Extracts:
        - Named entities
        - Noun chunks
        - Important nouns and verbs

        Args:
            text: Text to extract keywords from
            top_n: Maximum number of keywords to return

        Returns:
            List of keyword strings
        """
        try:
            doc = self.nlp(text.lower())
            keywords = set()

            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "WORK_OF_ART"]:
                    keywords.add(ent.text.strip())

            # Extract noun chunks
            for chunk in doc.noun_chunks:
                # Clean and filter noun chunks
                chunk_text = chunk.text.strip()
                if len(chunk_text) > 2 and not chunk_text.isspace():
                    keywords.add(chunk_text)

            # Extract important POSs (nouns, proper nouns, verbs)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop and len(token.text) > 2:
                    keywords.add(token.lemma_.strip())

            # Return top N most relevant keywords
            keyword_list = list(keywords)[:top_n]
            self.logger.debug(f"Extracted {len(keyword_list)} keywords from text")
            return keyword_list

        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def _compute_query_response_keyword_match( self, query: str, response: str) -> Dict[str, Any]:
        """
        Compute keyword match ratio between query and response.

        Args:
            query: User query text
            response: Agent response text

        Returns:
            Dictionary with match ratio and keyword details
        """
        try:
            query_keywords = set(self._extract_keywords(query))
            response_keywords = set(self._extract_keywords(response))
            if not query_keywords:
                return {
                    "match_ratio": 0.0,
                    "matched_keywords": [],
                    "query_keywords": [],
                    "response_keywords": list(response_keywords)
                }

            matched_keywords = query_keywords.intersection(response_keywords)
            match_ratio = len(matched_keywords) / len(query_keywords)

            result = {
                "match_ratio": round(match_ratio, 3),
                "matched_keywords": sorted(list(matched_keywords)),
                "query_keywords": sorted(list(query_keywords)),
                "response_keywords": sorted(list(response_keywords)),
                "num_query_keywords": len(query_keywords),
                "num_matched": len(matched_keywords)
            }
            self.logger.info(
                f"Q-R Keyword match: {len(matched_keywords)}/{len(query_keywords)} = {match_ratio:.1%}"
            )
            return result
        except Exception as e:
            self.logger.error(f"Error computing Q-R keyword match: {str(e)}")
            return {
                "match_ratio": 0.0,
                "matched_keywords": [],
                "query_keywords": [],
                "response_keywords": []
            }

    def _compute_source_response_keyword_match( self, source_documents: List[str], response: str) -> Dict[str, Any]:
        """
        Compute keyword match ratio between source documents and response.

        Args:
            source_documents: List of source document texts
            response: Agent response text

        Returns:
            Dictionary with match ratio and keyword details
        """
        try:
            # Extract keywords from all source documents
            all_source_keywords = set()
            for doc in source_documents:
                doc_keywords = self._extract_keywords(doc, top_n=15)
                all_source_keywords.update(doc_keywords)

            response_keywords = set(self._extract_keywords(response, top_n=15))
            if not all_source_keywords:
                return {
                    "match_ratio": 0.0,
                    "matched_keywords": [],
                    "source_keywords": [],
                    "response_keywords": list(response_keywords)
                }

            matched_keywords = all_source_keywords.intersection(response_keywords)
            match_ratio = len(matched_keywords) / len(all_source_keywords)

            result = {
                "match_ratio": round(match_ratio, 3),
                "matched_keywords": sorted(list(matched_keywords)),
                "source_keywords": sorted(list(all_source_keywords)),
                "response_keywords": sorted(list(response_keywords)),
                "num_source_keywords": len(all_source_keywords),
                "num_matched": len(matched_keywords)
            }

            self.logger.info(
                f"Source-R Keyword match: {len(matched_keywords)}/{len(all_source_keywords)} = {match_ratio:.1%}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error computing Source-R keyword match: {str(e)}")
            return {
                "match_ratio": 0.0,
                "matched_keywords": [],
                "source_keywords": [],
                "response_keywords": []
            }

    def _calculate_overall_score( self, qr_similarity: float, rs_max_similarity: float, qr_keyword_match: float, sr_keyword_match: float) -> float:
        """
        Calculate overall semantic evaluation score.

        Weighted combination of all metrics:
        - Query-Response similarity: 35%
        - Response-Source similarity: 35%
        - Q-R keyword match: 15%
        - S-R keyword match: 15%

        Args:
            qr_similarity: Query-response cosine similarity
            rs_max_similarity: Max response-source cosine similarity
            qr_keyword_match: Query-response keyword match ratio
            sr_keyword_match: Source-response keyword match ratio

        Returns:
            Overall score (0-1)
        """
        weights = {
            "qr_sim": 0.35,
            "rs_sim": 0.35,
            "qr_kw": 0.15,
            "sr_kw": 0.15
        }
        overall = (
            weights["qr_sim"] * qr_similarity +
            weights["rs_sim"] * rs_max_similarity +
            weights["qr_kw"] * qr_keyword_match +
            weights["sr_kw"] * sr_keyword_match
        )
        return round(overall, 3)

if __name__ == "__main__":
    # Initialize embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Initialize evaluator with the embedding model
    evaluator = SemanticSimilarityEvaluator(
        embedding_model=embedding_model,
        spacy_model="en_core_web_sm"
    )

    # Evaluate response
    result = evaluator.evaluate(
        user_query="How do I request vacation days?",
        agent_response="You can request vacation days through the HR portal by submitting a request at least two weeks in advance.",
        source_documents=[
            "Employees must submit vacation requests via the HR portal with at least two weeks' notice.",
            "The company policy requires advance notification for time off."
        ]
    )

    # Access results
    print(f"Query-Response Similarity: {result.query_response_similarity}")
    print(f"Response-Source Max Similarity: {result.response_source_max_similarity}")
    print(f"Q-R Keyword Match: {result.query_response_keyword_match}")
    print(f"Overall Score: {result.overall_semantic_score}")

