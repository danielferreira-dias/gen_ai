from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from utils.patterns import PIIType, PIIEntity
import re

load_dotenv()
class AzureLanguageService:
    def __init__(self):
        self.client = TextAnalyticsClient(
            endpoint=(os.getenv("AZURE_AI_LANGUAGE_ENDPOINT")), 
            credential=AzureKeyCredential(os.getenv("AZURE_AI_LANGUAGE_KEY"))
        )
    
    def pii_recognition_example(self, user_query : str ):
        try:
            documents = [f"{user_query}"]
            response = self.client.recognize_pii_entities(documents, language="en")
            result = response[0]

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
        
class NERService:
    def __init__(self, model_type: str = "bert"):
        self.model_type = model_type.lower()

        model_name = "dslim/bert-base-NER"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")

        # Map BERT-NER categories to your category format
        self.category_map = {
            "PER": "Person",
            "ORG": "Organization",
            "LOC": "Location",
            "MISC": "MISC"
        }

    def recognize_entities(self, user_query: str):
        try:
            return self._recognize_with_bert(user_query)
        except Exception as err:
            print(f"Encountered exception: {err}")
            return {
                'id': '0',
                'redacted_text': user_query,
                'is_error': True,
                'entities': []
            }

    def _recognize_with_bert(self, user_query: str):
        """Recognize entities using BERT"""
        # Run NER on the input text
        entities = self.ner_pipeline(user_query)

        # Create redacted text by replacing entities with their labels
        redacted_text = user_query

        # Sort entities by offset in reverse to avoid index shifting
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            mapped_category = self.category_map.get(entity['entity_group'], entity['entity_group'])
            redacted_text = redacted_text[:start] + f"[{mapped_category}]" + redacted_text[end:]

        # Format output similar to Azure PII service
        result_dict = {
            'id': '0',
            'redacted_text': redacted_text,
            'is_error': False,
            'entities': [
                {
                    'text': entity['word'],
                    'category': self.category_map.get(entity['entity_group'], entity['entity_group']),
                    'subcategory': None,
                    'offset': entity['start'],
                    'length': entity['end'] - entity['start'],
                    'confidence_score': float(entity['score'])
                }
                for entity in entities
            ]
        }

        return result_dict

class RegexService:
    def __init__(self):
        self.patterns = {
            PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            PIIType.PHONE: r'(?:' + '|'.join([
                r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\b(?:\+351\s?)?9[1236]\d{1}\s?\d{3}\s?\d{3}\b',
                r'\b(?:\+351\s?)?2[1-9]\d{1}\s?\d{3}\s?\d{3}\b',
                r'\b\(\+351\)\s?[29]\d{1}\s?\d{3}\s?\d{3}\b',
            ]) + r')',
            PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        }
    
    def detect(self, text: str) -> list[PIIEntity]:
        """Detect PII using regex patterns"""
        entities = []
        
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                # Additional validation for specific types
                if self._validate_match(match.group(), pii_type):
                    entities.append(PIIEntity(
                        text=match.group(),
                        pii_type=pii_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9
                    ))
        
        return entities

    def _validate_match(self, text: str, pii_type: PIIType) -> bool:
        """Validate regex matches with additional rules"""
        return True

class CustomPIIService:
    def __init__(self, model_type: str = "bert"):
        self.ner_service = NERService(model_type=model_type)
        self.regex_service = RegexService()

    def detect_pii(self, text: str) -> dict:
        # Get NER results
        ner_results = self.ner_service.recognize_entities(text)

        # Get Regex results
        regex_entities = self.regex_service.detect(text)

        # Convert regex entities to the same format as NER entities
        regex_formatted = [
            {
                'text': entity.text,
                'category': entity.pii_type.value,
                'subcategory': None,
                'offset': entity.start,
                'length': entity.end - entity.start,
                'confidence_score': entity.confidence
            }
            for entity in regex_entities
        ]

        # Combine both entity lists
        all_entities = ner_results['entities'] + regex_formatted

        # Sort by offset
        all_entities.sort(key=lambda x: x['offset'])

        # Create redacted text with all entities
        redacted_text = text
        sorted_entities = sorted(all_entities, key=lambda x: x['offset'], reverse=True)

        for entity in sorted_entities:
            start = entity['offset']
            end = entity['offset'] + entity['length']
            category = entity['category']
            redacted_text = redacted_text[:start] + f"[{category}]" + redacted_text[end:]

        return {
            'id': ner_results['id'],
            'original_text': text,
            'redacted_text': redacted_text,
            'is_error': ner_results['is_error'],
            'ner_entities': ner_results['entities'],
            'regex_entities': regex_formatted,
            'all_entities': all_entities,
            'entity_count': {
                'ner': len(ner_results['entities']),
                'regex': len(regex_formatted),
                'total': len(all_entities)
            }
        }
