from services.agent import Agent
from services.pii import AzureLanguageService
from dotenv import load_dotenv
import asyncio
from services.processing import ProcessingService

load_dotenv()

async def main():
    # agent = Agent(model_name="gpt-5-chat")  
    # user_query = "Hey, what's your job?"
    # print(f"User: {user_query}")
    # response = await agent.llm_response(user_query)
    # print(f"Agent: {response}")
    azure_service = AzureLanguageService()
    data = azure_service.pii_recognition_example()
    print('=' * 60)
    print(f'PII -> {data}')
    print('=' * 60)

    processing_Service = ProcessingService()
    process_data = processing_Service.tokenize_pii(data=data)
    print('=' * 60)
    print(f'Tokenized Data -> {process_data}')
    print('=' * 60)

if __name__ == "__main__":
    asyncio.run(main())
