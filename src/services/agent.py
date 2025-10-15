import logging
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dataclasses import dataclass
from langchain.agents import create_agent


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
        self.system_prompt = "You're an internal company's model that's helping answer user queries, always mention the user's name in the beginning."
        self.agent = create_agent(
            model=self.model,
            tools=[],
            response_format=AgentOutput,
        )

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
    

