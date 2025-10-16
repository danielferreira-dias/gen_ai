import logging
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dataclasses import dataclass
from langchain.agents import create_agent
from .chroma_service import ChromaService, EmbeddingModel
from langchain_core.tools import tool


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class AgentOutput:
    response: str

@dataclass
class AgentResponseWithMetadata:
    """Contains agent response and tool invocation metadata"""
    response: str
    tool_calls: list  # List of tool invocations with details

class Agent:

    def __init__(self, model_name: str, chroma_service: ChromaService = None):
        self.model = AzureChatOpenAI(
            azure_deployment=model_name,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=1,
        )

        # Use provided ChromaDB service or create a new one
        if chroma_service is None:
            embedding_model = EmbeddingModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
            self.chroma_service = ChromaService(embedding_model=embedding_model)
        else:
            self.chroma_service = chroma_service

        # Store tool invocations for this conversation turn
        self.tool_invocations = []

        self.tools = self.create_tools()
        self.system_prompt = """You're an internal company's model that's helping answer user queries, leveraging Porto's knowledge base when needed. Always provide accurate and concise information.

IMPORTANT: You have access to a tool called 'get_knowledge_base_answer' that searches the company's knowledge base.
- When a user asks a question that requires information from the knowledge base, you MUST use this tool.
- The tool requires a 'user_query' parameter - pass the user's question or a relevant search query as this parameter.
- Example: If user asks "What are the best restaurants in Porto?", call get_knowledge_base_answer(user_query="What are the best restaurants in Porto?")
- Always use the tool when users ask about company information, Porto locations, recommendations, or specific factual questions."""
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            response_format=AgentOutput,
        )

    def create_tools(self):
        @tool("get_knowledge_base_answer", description="Use this tool to answer user queries based on the company's knowledge base. If you cannot find relevant information, respond with 'No relevant information found in the knowledge base.'")
        async def get_knowledge_base_answer(user_query: str) -> str:
            logger.info("=" * 50)
            logger.info(f"TOOL CALLED: get_knowledge_base_answer")
            logger.info(f"Query: {user_query}")
            logger.info("=" * 50)

            try:
                # Search for relevant documents with scores
                logger.info(f"Starting knowledge base search for: {user_query}")
                results = await self.chroma_service.search_with_scores(user_query, k=3)
                logger.info(f"Search completed! Retrieved {len(results)} results")

                # Log the actual results
                for i, result in enumerate(results, 1):
                    logger.info(f"Result {i}: Score={result.get('Score', 'N/A')}, Source={result.get('Source', 'N/A')}")

                # Store tool invocation metadata
                self.tool_invocations.append({
                    'tool_name': 'get_knowledge_base_answer',
                    'query': user_query,
                    'results_count': len(results),
                    'results': results
                })
                logger.info(f"Tool invocation metadata stored. Total invocations: {len(self.tool_invocations)}")

                if not results:
                    logger.warning("No results found - returning 'not found' message")
                    return "No relevant information found in the knowledge base."

                # Format the results into a readable response
                response_parts = ["Here's what I found in the knowledge base:\n"]

                for i, result in enumerate(results, 1):
                    response_parts.append(f"\n{i}. {result['Content'][:300]}...")
                    response_parts.append(f"   (Source: {result['Source']}, Relevance Score: {result['Score']:.4f})")

                final_response = "\n".join(response_parts)
                logger.info(f"Returning formatted response (length: {len(final_response)} chars)")
                logger.info("=" * 50)
                return final_response

            except Exception as e:
                logger.error(f"ERROR in get_knowledge_base_answer: {e}", exc_info=True)
                logger.error("=" * 50)
                return f"Error accessing knowledge base: {str(e)}"

        return [get_knowledge_base_answer]

    async def llm_response(self, user_query: str, conversation_history: list = None):
        # Clear tool invocations from previous turn
        self.tool_invocations = []
        logger.info(f"Starting llm_response for query: {user_query[:100]}...")

        messages = [SystemMessage(content=self.system_prompt)]
        logger.info(f'Conversation History: {len(conversation_history) if conversation_history else 0} messages')
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                # Handle dictionary format from database
                if isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content')
                    if role == 'user':
                        messages.append(HumanMessage(content=content))
                    elif role == 'assistant':
                        messages.append(AIMessage(content=content))
                    elif role == 'system':
                        messages.append(SystemMessage(content=content))

        # Add current user query
        messages.append(HumanMessage(content=user_query))

        # Invoke agent with full context
        logger.info(f'Invoking agent with {len(messages)} messages')
        response = await self.agent.ainvoke({"messages": messages})

        logger.info(f"Agent response received. Type: {type(response)}")
        logger.info(f"Agent response keys: {response.keys() if isinstance(response, dict) else 'Not a dict'}")

        final_response = response.get('structured_response').response
        if final_response is None:
            final_response = "No Response"

        logger.info(f"Final response: {final_response[:200]}...")
        logger.info(f"Tool invocations recorded: {len(self.tool_invocations)}")

        # Return response with tool invocation metadata
        return AgentResponseWithMetadata(
            response=final_response,
            tool_calls=self.tool_invocations.copy()
        )
    

