from services.agent_service import Agent
from services.pii_service import AzureLanguageService, NERService
from dotenv import load_dotenv
import asyncio
from services.process_service import ProcessingService

load_dotenv()

async def main():
    ner = NERService(model_type="bert", spacy_model="")
    response = ner.recognize_entities("I'm Daniel Dias, I live in Rua Principal São Félix, can you tell me what are some of the best francesinhas to eat in Porto")
    print(('=' * 30) + 'NER Response' + ('=' * 30))

    proccess = ProcessingService()
    response = proccess.tokenize_pii(data=response)

    print(('=' * 30) + 'AFTER process' + ('=' * 30))
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
