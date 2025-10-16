from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from utils.prompts import llm_judge_prompt
from database.evaluation import EvaluationStorage
import logging
import os
import re
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

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

    def calculate_heuristics(
        self,
        agent_response: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
        response_time_ms: Optional[float] = None
    ) -> Dict[str, Any]:
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

    def _calculate_retrieval_confidence(
        self,
        retrieved_docs: Optional[List[Dict[str, Any]]]
    ) -> float:
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

