class ProcessingService:
    def __init__(self):
        self.category_counters = {}
        self.token_map = {}
        self.category_map = {
            "PhoneNumber": "NUMBER",
            "Address": "LOCATION_OFFICE",
            "Person": "PERSON",
            "Address":"ADDRESS",
            "Organization":"ORGANIZATION",
            "City":"CITY",
            "Location":"LOCATION",
            "Email":"EMAIL",
        }

    def tokenize_pii(self, data: dict, existing_token_map: dict = None):
        """
        Tokenize PII entities in the text with placeholders like <PERSON_1>, <NUMBER_1>, etc.
        Args:
            data: Dictionary containing 'id', 'redacted_text', 'entities', etc.
            existing_token_map: Existing token map from the conversation to prevent token collisions
        Returns:
            Dictionary with tokenized text, token mapping, and has_pii flag
        """

        if data is None:
            return None

        # Initialize counters based on existing token map to avoid collisions
        self.category_counters = {}
        self.token_map = {}

        if existing_token_map:
            # Parse existing tokens to find the highest counter for each category
            for token in existing_token_map.keys():
                # Token format: <CATEGORY_NUMBER>
                if token.startswith('<') and token.endswith('>'):
                    token_content = token[1:-1]  # Remove < >
                    parts = token_content.rsplit('_', 1)  # Split from the right to get category and number
                    if len(parts) == 2:
                        category, num_str = parts
                        try:
                            num = int(num_str)
                            # Set counter to the max seen for this category
                            if category not in self.category_counters:
                                self.category_counters[category] = num
                            else:
                                self.category_counters[category] = max(self.category_counters[category], num)
                        except ValueError:
                            pass  # Skip if number parsing fails

        # Get the original text from entities (reconstruct from redacted text)
        entities = data.get('entities', [])

        # Check if there are any PII entities
        has_pii = len(entities) > 0

        # If no PII, return original redacted_text as-is
        if not has_pii:
            return {
                'original_id': data.get('id'),
                'tokenized_text': data.get('redacted_text', ''),
                'token_map': {},
                'entities': [],
                'has_pii': False
            }

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

            # Special handling for specific categories
            category_token = self.category_map.get(category, category.upper())

            # Increment counter for this category
            if category_token not in self.category_counters:
                self.category_counters[category_token] = 0

            self.category_counters[category_token] += 1
            count = self.category_counters[category_token]

            # Create token placeholder
            token = f"<{category_token}_{count}>"

            # Store mapping for detokenization
            self.token_map[token] = text

            tokenized_text = tokenized_text[:offset] + token + tokenized_text[offset + length:]

        return {
            'original_id': data.get('id'),
            'tokenized_text': tokenized_text,
            'token_map': self.token_map,
            'entities': entities,
            'has_pii': True
        }

    def detokenize_pii(self, text: str, token_map: dict = None):
        """
        Replace tokens back with original PII values.

        Args:
            text: Text containing tokens (e.g., agent response)
            token_map: Dictionary mapping tokens to original values (uses self.token_map if not provided)

        Returns:
            Text with PII restored
        """
        if token_map is None:
            token_map = self.token_map

        # If no token map, return text as-is
        if not token_map:
            return text

        detokenized_text = text

        # Replace each token with its original value
        for token, original_value in token_map.items():
            detokenized_text = detokenized_text.replace(token, original_value)

        return detokenized_text