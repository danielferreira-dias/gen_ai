from services.agent import Agent
from services.pii import AzureLanguageService, NERService
from dotenv import load_dotenv
import asyncio
from services.processing import ProcessingService

load_dotenv()

async def main():
    # azure_service = AzureLanguageService()
    # data = azure_service.pii_recognition_example()
    # print(('=' * 30) + 'PII' + ('=' * 30))
    # print(f'{data}')
    # print('=' * 60)

    # processing_Service = ProcessingService()
    # process_data = processing_Service.tokenize_pii(data=data)
    # print(('=' * 30) + 'Tokenized Data' + ('=' * 30))
    # print(f'{process_data}')
    # print('=' * 60)

    # # Check if PII was found
    # if process_data.get('has_pii'):
    #     print("PII detected - using tokenized text for LLM")
    # else:
    #     print("No PII detected - using original text for LLM")

    # # Send tokenized text to agent (without conversation history for this demo)
    # agent = Agent(model_name="gpt-5-chat")
    # response = await agent.llm_response(
    #     user_query=process_data.get('tokenized_text'),
    #     conversation_history=None  # No history for single-message demo
    # )
    # print(('=' * 30) + 'Agent Response (Tokenized)' + ('=' * 30))
    # print(f'{response}')
    # print('=' * 60)

    # # De-tokenize the agent's response to restore PII
    # if process_data.get('has_pii'):
    #     detokenized_response = processing_Service.detokenize_pii(
    #         text=response,
    #         token_map=process_data.get('token_map')
    #     )
    #     print(('=' * 30) + 'Agent Response (De-tokenized)' + ('=' * 30))
    #     print(f'{detokenized_response}')
    #     print('=' * 60)
    ner = NERService()
    response = ner.recognize_entities("My name is Daniel Dias, I work at LudusCristalTec in Porto, in Rua dos Engenheiros, yesterday I spoke with Maria Junah")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
