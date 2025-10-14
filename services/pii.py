from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os 
from dotenv import load_dotenv

load_dotenv()

class AzureLanguageService:
    def __init__(self):
        self.client = TextAnalyticsClient(
            endpoint=(os.getenv("AZURE_AI_LANGUAGE_ENDPOINT")), 
            credential=AzureKeyCredential(os.getenv("AZURE_AI_LANGUAGE_KEY"))
        )
        
    # Example method for detecting sensitive information (PII) from text 
    def pii_recognition_example(self):
        documents = [
            "The employee's SSN is 859-98-0987.",
            "The employee's phone number is 555-555-5555."
        ]
        return self.client.recognize_pii_entities(documents, language="en")
        
