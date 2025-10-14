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
    print(('=' * 30) + 'PII' + ('=' * 30))
    print(f'{data}')
    print('=' * 60)

    processing_Service = ProcessingService()
    process_data = processing_Service.tokenize_pii(data=data)
    print(('=' * 30) + 'Tokenized Data' + ('=' * 30))
    print(f'{process_data}')
    print('=' * 60)

    agent = Agent(model_name="gpt-5-chat")  
    response = await agent.llm_response(process_data.get('tokenized_text'))
    print(('=' * 30) + 'Agent Response' + ('=' * 30))
    print(f'{response}')
    print('=' * 60)

if __name__ == "__main__":
    asyncio.run(main())
