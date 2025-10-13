import os
from dotenv import load_dotenv
import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMesssage
from dataclasses import dataclass

load_dotenv()

@dataclass
class AgentOutput:
    response: str

class LLM:
    def __init__(self, model_name: str):
        self.model = AzureChatOpenAI(
            azure_deployment=model_name,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=1,
        )

        self.system_prompt = "You're an internal company's model that's helping answer user queries"

    async def response(self, user_query: str):
        response = await self.agent.ainvoke({"messages": [ SystemMessage(f"{self.system_prompt}") ,HumanMessage(content=user_query)]})
        final_response = response.get('structured_response').response
        if final_response is None:
            return "No Response" 
        return final_response