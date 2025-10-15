from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
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
    def __init__(self, model_type: str = "spacy", spacy_model: str = "en_core_web_sm"):
        """
        Initialize NER Service with either spaCy or BERT model.

        Args:
            model_type: Either "spacy" or "bert"
            spacy_model: Name of spaCy model to use (e.g., "en_core_web_sm", "pt_core_news_sm" for Portuguese)
        """
        self.model_type = model_type.lower()

        if self.model_type == "spacy":
            # Load spaCy model
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"spaCy model '{spacy_model}' not found. Run: python -m spacy download {spacy_model}")
                raise

            # Map spaCy entity types to your category format
            self.category_map = {
                "PERSON": "Person",
                "PER": "Person",
                "ORG": "Organization",
                "GPE": "Location",  # Geopolitical entity
                "LOC": "Location",
                "MISC": "MISC"
            }
        elif self.model_type == "bert":
            # Load BERT model
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
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Choose 'spacy' or 'bert'")

    def recognize_entities(self, user_query: str):
        try:
            if self.model_type == "spacy":
                return self._recognize_with_spacy(user_query)
            else:
                return self._recognize_with_bert(user_query)

        except Exception as err:
            print(f"Encountered exception: {err}")
            return {
                'id': '0',
                'redacted_text': user_query,
                'is_error': True,
                'entities': []
            }

    def _recognize_with_spacy(self, user_query: str):
        """Recognize entities using spaCy"""
        doc = self.nlp(user_query)

        # Extract entities
        entities = []
        for ent in doc.ents:
            mapped_category = self.category_map.get(ent.label_, ent.label_)
            entities.append({
                'text': ent.text,
                'category': mapped_category,
                'start': ent.start_char,
                'end': ent.end_char,
                'label': ent.label_
            })

        # Create redacted text by replacing entities with their labels
        redacted_text = user_query
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            redacted_text = redacted_text[:start] + f"[{entity['category']}]" + redacted_text[end:]

        # Format output similar to Azure PII service
        result_dict = {
            'id': '0',
            'redacted_text': redacted_text,
            'is_error': False,
            'entities': [
                {
                    'text': entity['text'],
                    'category': entity['category'],
                    'subcategory': None,
                    'offset': entity['start'],
                    'length': entity['end'] - entity['start'],
                    'confidence_score': 0.85  # spaCy doesn't provide confidence scores by default
                }
                for entity in entities
            ]
        }

        return result_dict

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
                # US phone formats
                r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                # Portuguese mobile: +351 9XX XXX XXX or 9XX XXX XXX
                r'\b(?:\+351\s?)?9[1236]\d{1}\s?\d{3}\s?\d{3}\b',
                # Portuguese landline: +351 2XX XXX XXX or 2XX XXX XXX
                r'\b(?:\+351\s?)?2[1-9]\d{1}\s?\d{3}\s?\d{3}\b',
                # Portuguese with parentheses: (+351) 9XX XXX XXX
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
