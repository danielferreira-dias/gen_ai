from services.agent_service import Agent
from services.pii_service import AzureLanguageService, NERService, CustomPIIService
from dotenv import load_dotenv
import asyncio
from services.process_service import ProcessingService
import json

load_dotenv()

async def main():
    # Test text with multiple types of PII
    test_text = "I'm Daniel Dias, I live in Rua Principal São Félix. You can reach me at daniel@example.com or call me at +351 912 345 678. My SSN is 123-45-6789."

    print(('=' * 30) + ' CUSTOM PII SERVICE TEST ' + ('=' * 30))
    print(f"Original text: {test_text}\n")

    # Initialize and test CustomPIIService
    custom_pii = CustomPIIService(model_type="bert")
    response = custom_pii.detect_pii(test_text)

    print(f"Redacted text: {response['redacted_text']}\n")
    print(f"Entity counts: {json.dumps(response['entity_count'], indent=2)}\n")

    print("NER Entities:")
    for entity in response['ner_entities']:
        print(f"  - {entity['text']:<25} | {entity['category']:<15} | Confidence: {entity['confidence_score']:.3f}")

    print("\nRegex Entities:")
    for entity in response['regex_entities']:
        print(f"  - {entity['text']:<25} | {entity['category']:<15} | Confidence: {entity['confidence_score']:.3f}")

    print("\n" + ('=' * 30) + ' PROCESSING SERVICE TEST ' + ('=' * 30))

    proccess = ProcessingService()
    processed_response = proccess.tokenize_pii(data=response)

    print(('=' * 30) + 'AFTER process' + ('=' * 30))
    print(processed_response)

if __name__ == "__main__":
    asyncio.run(main())
