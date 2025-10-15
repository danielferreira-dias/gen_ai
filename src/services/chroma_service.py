import logging
import os
import asyncio
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()


class EmbeddingModel:
    """Wrapper for HuggingFace embedding models"""
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.embeddings = HuggingFaceEmbeddings(model_name=model_id)
        logger.info(f'✓ Loaded embedding model: {model_id}')


class ChromaService:
    """Service for vector similarity search using ChromaDB"""

    def __init__(self, embedding_model: EmbeddingModel, persist_directory: str = None):
        """
        Initialize ChromaDB service

        Args:
            embedding_model: EmbeddingModel instance
            persist_directory: Path to ChromaDB storage (defaults to src/db/chroma.sqlite3)
        """
        self.embedding_model = embedding_model

        # Default to the database location you specified
        if persist_directory is None:
            persist_directory = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'db'
            )

        self.vectorstore = Chroma(
            collection_name="fraud_analysis_docs",
            embedding_function=embedding_model.embeddings,
            persist_directory=persist_directory
        )
        logger.info(f'✓ Connected to ChromaDB at {persist_directory}')

    async def search_documents(self, user_query: str, k: int = 5):
        """
        Async vector similarity search

        Args:
            user_query: Query string to search for
            k: Number of results to return

        Returns:
            List of dictionaries containing search results
        """
        results = await self.vectorstore.asimilarity_search(query=user_query, k=k)

        return [{
            'Content': result.page_content,
            'Source': result.metadata.get('source', 'Unknown'),
            'Metadata': result.metadata
        } for result in results]

    async def search_with_scores(self, user_query: str, k: int = 5):
        """
        Async vector similarity search with relevance scores

        Args:
            user_query: Query string to search for
            k: Number of results to return

        Returns:
            List of dictionaries containing search results with scores
        """
        results_with_scores = await self.vectorstore.asimilarity_search_with_score(
            query=user_query,
            k=k
        )

        return [{
            'Content': result.page_content,
            'Source': result.metadata.get('source', 'Unknown'),
            'Metadata': result.metadata,
            'Score': score
        } for result, score in results_with_scores]


# Usage example
async def main():
    """Example usage of ChromaService"""
    embedding_model = EmbeddingModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
    service = ChromaService(embedding_model=embedding_model)

    # Vector search
    print("\n--- Vector Search Results ---")
    results = await service.search_documents("What are the best francesinhas in Porto?", k=3)

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Content: {result['Content'][:200]}...")
        print(f"Source: {result['Source']}")

    # Search with scores
    print("\n\n--- Vector Search with Scores ---")
    results_with_scores = await service.search_with_scores("What are the best francesinhas in Porto? I'm Daniel Dias", k=3)

    for i, result in enumerate(results_with_scores, 1):
        print(f"\nResult {i} (Score: {result['Score']:.4f}):")
        print(f"Content: {result['Content'][:200]}...")
        print(f"Source: {result['Source']}")


if __name__ == "__main__":
    asyncio.run(main())
