from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os 
import json
from dotenv import load_dotenv

load_dotenv()

class AzureLanguageService:
    def __init__(self):
        self.client = TextAnalyticsClient(
            endpoint=(os.getenv("AZURE_AI_LANGUAGE_ENDPOINT")), 
            credential=AzureKeyCredential(os.getenv("AZURE_AI_LANGUAGE_KEY"))
        )
    
    def pii_recognition_example(self):
        try:
            documents = ["I had a wonderful trip to Seattle last week. My number is 915518582, my name is John Doe from the office in Lisbon, 21th Street Market"]
            response = self.client.recognize_pii_entities(documents, language="en")
            result = response[0]

            # Convert result to dict
            result_dict = {
                'id': result.id,
                'redacted_text': result.redacted_text,
                'is_error': result.is_error,
                'entities': [
                    {
                        'text': entity.text,
                        'category': entity.category,
                        'subcategory': getattr(entity, 'subcategory', None),
                        'offset': entity.offset,
                        'length': entity.length,
                        'confidence_score': entity.confidence_score
                    }
                    for entity in result.entities
                ]
            }

            return result_dict

        except Exception as err:
            print("Encountered exception. {}".format(err))
        
class CustomPIIService:
    def __init__(self):
        pass
        
