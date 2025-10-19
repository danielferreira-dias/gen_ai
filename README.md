# PII-Aware LLM Application with Production Evaluation

A production-ready LLM application that implements comprehensive PII (Personally Identifiable Information) protection and multi-layered evaluation strategies to ensure safe, high-quality responses in production environments.

[![Watch the video](https://img.youtube.com/vi/TVunaf6hkY8/0.jpg)](https://youtu.be/TVunaf6hkY8?si=m9UHXbRZ2vGqKv0L)

## Project Overview

This project addresses two critical challenges in deploying LLM applications:

1. **PII Risk Mitigation**: Prevents sensitive user information from being exposed to LLM providers through intelligent detection and tokenization
2. **Production Evaluation**: Implements multiple evaluation strategies to continuously monitor and improve model quality in production

## Architecture

### 1. PII Detection Layer

<img width="2880" height="1288" alt="Privacy_Risk" src="https://github.com/user-attachments/assets/105f18db-2bf4-4fb9-8d1b-c144f99b71fb" />

The PII Detection Layer acts as a protective shield between users and the LLM:

**Components:**
- **PII Service**: Azure AI Language Service integration for entity recognition
- **Pre-Processing Service**: Tokenizes detected PII entities before LLM processing
- **Post-Processing Service**: De-tokenizes responses to restore original user context
- **GuardRails**: Validates inputs and outputs for safety and compliance

**Workflow:**
1. User submits a query containing PII (e.g., names, addresses, phone numbers)
2. PII Service detects and categorizes sensitive entities
3. Pre-Processing Service replaces PII with secure tokens (e.g., `[NAME_1]`, `[EMAIL_1]`)
4. Tokenized query is sent to the LLM
5. LLM processes the query using tokenized placeholders
6. Post-Processing Service restores original PII in the response
7. User receives a contextually accurate response without exposing PII to the LLM

### 2. Continuous Evaluation (Concept)

<img width="2828" height="2620" alt="Privacy_Risk_Evaluation" src="https://github.com/user-attachments/assets/fd805b78-73d3-43e0-9832-fef4075b3cf3" />

**Note**: This is a conceptual design for future implementation to enable periodic evaluation of the LLM in production.

The evaluation pipeline will support multiple strategies:

**Evaluation Methods:**
- **Ragas / Azure AI Foundry Evaluators**: Pre-built evaluation frameworks for RAG systems
- **Custom Service**: Domain-specific evaluation logic
- **LLM as a Judge**: Using a separate LLM to assess response quality
- **Human in the Loop**: Expert review for edge cases and quality assurance
- **Regression Testing**: Automated testing against known good responses
- **A/B Testing**: Compare different model versions or configurations

**Ground Truth Dataset:**
- Synthetic data generation
- Expert-curated question-answer pairs with expected sources
- Continuous expansion based on production patterns

**Metrics Tracked:**
- Groundedness: Are responses based on retrieved documents?
- Relevance: Do responses address the user's question?
- Similarity: Semantic alignment between query and response
- Retrieval Accuracy: Quality of document retrieval

### 3. A/B Testing Infrastructure

<img width="2948" height="1438" alt="AB_Testing" src="https://github.com/user-attachments/assets/3e78c185-7aed-4edb-9411-b688796aca41" />

**Note**: This is a conceptual design for comparing different agent versions in production.

**Components:**
- **API Gateway**: Routes traffic to different agent versions
- **Load Balancer**: Distributes requests between Agent v0 and Agent v1
- **Evaluation Layer**: Compares performance metrics across versions
- **Database**: Stores comparative results and metrics

## Currently Implemented Features

### PII Protection
- Azure AI Language Service integration for PII detection
- Multi-entity support (names, emails, phone numbers, addresses, etc.)
- Bidirectional tokenization (pre and post-processing)
- Conversation-level token map management
- PII audit logging

### RAG (Retrieval-Augmented Generation)
- ChromaDB vector database integration
- Sentence transformers for document embedding
- Semantic search with configurable similarity thresholds
- Document ingestion pipeline

### Evaluation
- **Lightweight Heuristics**: Fast, rule-based quality checks
  - Response length validation
  - Citation presence detection
  - Retrieval confidence scoring
  - Response time monitoring

- **LLM as a Judge**: Comprehensive response evaluation
  - Relevance scoring
  - Accuracy assessment
  - PII violation detection
  - Safety violation checking
  - Clarity evaluation

- **Semantic Similarity**: Vector-based quality metrics
  - Query-response similarity
  - Response-source alignment
  - Keyword matching (using spaCy NLP)

- **User Feedback**: Thumbs up/down rating system

### Observability
- Structured logging with timestamps
- Conversation tracking
- Message-level metadata storage
- Tool invocation tracing
- Performance metrics collection

### Storage
- SQLite database for conversations and evaluations
- ChromaDB for vector embeddings
- Token map persistence
- Audit log storage

## Technical Stack

- **Framework**: Streamlit (Interactive UI)
- **LLM Provider**: Azure OpenAI (GPT-4)
- **PII Detection**: Azure AI Language Service
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **NLP**: spaCy (en_core_web_sm)
- **Storage**: SQLite
- **Language**: Python 3.11+

## Project Structure

```
gen_ai/
├── src/
│   ├── app.py                      # Main Streamlit application
│   ├── main.py                     # Alternative entry point
│   ├── database/
│   │   ├── storage.py              # Conversation and message storage
│   │   └── evaluation.py           # Evaluation metrics storage
│   ├── services/
│   │   ├── pii_service.py          # Azure AI Language integration
│   │   ├── process_service.py      # PII tokenization/de-tokenization
│   │   ├── agent_service.py        # LLM agent orchestration
│   │   ├── chroma_service.py       # Vector database operations
│   │   └── evaluation_service.py   # Multi-strategy evaluation
│   └── utils/
│       ├── prompts.py              # LLM prompt templates
│       └── patterns.py             # Regex patterns for PII
├── ingestion/
│   └── main.py                     # Document ingestion pipeline
├── tests/                          # Unit tests
└── db/                             # SQLite and ChromaDB storage
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Azure OpenAI API access
- Azure AI Language Service credentials

### Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Configuration

Set up your environment variables:

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_LANGUAGE_KEY="your-language-key"
export AZURE_LANGUAGE_ENDPOINT="your-language-endpoint"
```

### Running the Application

```bash
# Run the Streamlit app
streamlit run src/app.py

# Or use the alternative entry point
python src/main.py
```

### Ingesting Documents

```bash
# Add documents to the vector database
python ingestion/main.py
```

## Usage

1. **Start a conversation**: Type your message in the chat input
2. **Enable debug mode**: Toggle "Enable Debug Tracing" in the sidebar to see:
   - PII detection and tokenization results
   - Tool invocations and vector search results
   - Lightweight heuristics
   - LLM Judge evaluation
   - Semantic similarity scores
3. **Provide feedback**: Use thumbs up/down on assistant responses
4. **View PII audit**: Click "View PII Log" in the sidebar to see detected entities

## Key Features in Detail

### PII Tokenization Flow

```python
# User input: "My name is John Doe and my email is john@example.com"
# Detected PII: [{"text": "John Doe", "category": "Person"}, {"text": "john@example.com", "category": "Email"}]
# Tokenized: "My name is [NAME_1] and my email is [EMAIL_1]"
# LLM sees only: "My name is [NAME_1] and my email is [EMAIL_1]"
# Response: "Hello [NAME_1], I've noted your email as [EMAIL_1]"
# De-tokenized: "Hello John Doe, I've noted your email as john@example.com"
```

### Multi-Layer Evaluation

Each response is evaluated through:

1. **Lightweight Heuristics** (~5ms): Fast quality gates
2. **Semantic Similarity** (~50ms): Vector-based relevance
3. **LLM Judge** (~2s): Comprehensive quality assessment
4. **User Feedback**: Human-in-the-loop validation

## Future Roadmap

- [ ] Implement continuous evaluation pipeline with ground truth datasets
- [ ] Add A/B testing infrastructure
- [ ] Support for regression testing
- [ ] Integration with Azure AI Foundry Evaluators
- [ ] Advanced observability with distributed tracing
- [ ] Multi-model support and fallback strategies
- [ ] Real-time alerting for quality degradation

## Security Considerations

- PII is never stored in plain text in vector databases
- Token maps are scoped to conversations
- Audit logs track all PII detection events
- Safety violations are automatically flagged
- Regular evaluation prevents model drift

## Contributing

Contributions are welcome! Please ensure:
- All new features include appropriate tests
- PII protection is maintained in new code paths
- Evaluation metrics are updated for new capabilities

## License

[Add your license here]

## Acknowledgments

- Azure AI Language Service for PII detection
- ChromaDB for vector storage
- LangSmith for observability patterns
- Ragas for evaluation framework inspiration
