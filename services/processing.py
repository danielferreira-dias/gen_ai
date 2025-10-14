class ProcessingService:
    def __init__(self):
        self.category_counters = {}
        self.token_map = {}
        self.category_map = {
            "PhoneNumber": "NUMBER",
            "Address": "LOCATION_OFFICE",
            # Add other special mappings here if needed
        }

    def tokenize_pii(self, data: dict):
        """
        Tokenize PII entities in the text with placeholders like <PERSON_1>, <NUMBER_1>, etc.
        Args:
            data: Dictionary containing 'id', 'redacted_text', 'entities', etc.
        Returns:
            Dictionary with tokenized text and token mapping
        """

        if data is None:
            return
        # Reset counters for this document
        self.category_counters = {}
        self.token_map = {}
        # Get the original text from entities (reconstruct from redacted text)
        entities = data.get('entities', [])
        # Sort entities by offset in reverse order to replace from end to start
        # This prevents offset shifting issues
        sorted_entities = sorted(entities, key=lambda x: x['offset'], reverse=True)
        tokenized_text = data.get('redacted_text', '')

        # Process each entity
        for entity in sorted_entities:
            category = entity['category']
            text = entity['text']
            offset = entity['offset']
            length = entity['length']

            
            category_upper = category.upper()
            # Special handling for specific categories
            if category == 'PhoneNumber':
                category_token = 'NUMBER'
            elif category == 'Address':
                category_token = 'LOCATION_OFFICE'
            else:
                category_token = category_upper

            # Increment counter for this category
            if category_token not in self.category_counters:
                self.category_counters[category_token] = 0

            self.category_counters[category_token] += 1
            count = self.category_counters[category_token]

            # Create token placeholder
            token = f"<{category_token}_{count}>"

            # Store mapping for detokenization
            self.token_map[token] = text

            # Replace in text (using offset and length)
            # Note: redacted_text has asterisks, we need to replace those
            # Find the asterisk sequence or the actual text
            # Since we're working backwards, offsets remain valid
            tokenized_text = tokenized_text[:offset] + token + tokenized_text[offset + length:]

        return {
            'original_id': data.get('id'),
            'tokenized_text': tokenized_text,
            'token_map': self.token_map,
            'entities': entities
        }

    def detokenize_pii(self, data: dict):
        """
            Replace tokens back with original PII values.
        Args:
            data: Dictionary containing 'tokenized_text' and 'token_map'
        Returns:
            Original text with PII restored
        """
        tokenized_text = data.get('tokenized_text', '')
        token_map = data.get('token_map', {})

        detokenized_text = tokenized_text

        # Replace each token with its original value
        for token, original_value in token_map.items():
            detokenized_text = detokenized_text.replace(token, original_value)

        return {
            'detokenized_text': detokenized_text,
            'original_id': data.get('original_id')
        }