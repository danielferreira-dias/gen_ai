from services.agent import Agent
from services.pii import AzureLanguageService
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    # agent = Agent(model_name="gpt-5-chat")  
    # user_query = "Hey, what's your job?"
    # print(f"User: {user_query}")
    # response = await agent.llm_response(user_query)
    # print(f"Agent: {response}")

    azure_service = AzureLanguageService()
    data = azure_service.pii_recognition_example()
    print(f'Data -> {data}')

if __name__ == "__main__":
    asyncio.run(main())
