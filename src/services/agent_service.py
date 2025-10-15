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

class Agent:

    def __init__(self, model_name: str):
        self.model = AzureChatOpenAI(
            azure_deployment=model_name,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=1,
        )

        # Initialize ChromaDB service for knowledge base retrieval
        embedding_model = EmbeddingModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
        self.chroma_service = ChromaService(embedding_model=embedding_model)

        self.tools = self.create_tools()
        self.system_prompt = "You're an internal company's model that's helping answer user queries, leveraging Porto's knowledge base when needed. Always provide accurate and concise information."
        self.agent = create_agent(
            model=self.model,
            tools=self.tools,
            response_format=AgentOutput,
        )

    def create_tools(self):
        @tool("get_knowledge_base_answer", description="Use this tool to answer user queries based on the company's knowledge base. If you cannot find relevant information, respond with 'No relevant information found in the knowledge base.'")
        async def get_knowledge_base_answer(user_query: str) -> str:
            try:
                # Search for relevant documents
                print("user_query", user_query)
                print("Retrieving from knowledge base...")
                results = await self.chroma_service.search_documents(user_query, k=3)
                print("Results:", results)

                if not results:
                    return "No relevant information found in the knowledge base."

                # Format the results into a readable response
                response_parts = ["Here's what I found in the knowledge base:\n"]

                for i, result in enumerate(results, 1):
                    response_parts.append(f"\n{i}. {result['Content'][:300]}...")
                    response_parts.append(f"(Source: {result['Source']})")

                return "\n".join(response_parts)

            except Exception as e:
                logger.error(f"Error retrieving from knowledge base: {e}")
                return f"Error accessing knowledge base: {str(e)}"

        return [get_knowledge_base_answer]

    async def llm_response(self, user_query: str, conversation_history: list = None):
        messages = [SystemMessage(content=self.system_prompt)]
        logger.info(f'Conversation History {conversation_history}')
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
        logger.info(f'Current Messages being fed {messages}')
        response = await self.agent.ainvoke({"messages": messages})
        
        final_response = response.get('structured_response').response
        if final_response is None:
            return "No Response"
        
        return final_response
    

