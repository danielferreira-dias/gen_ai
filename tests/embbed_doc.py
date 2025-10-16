from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

# Evaluation prompt template
EVAL_PROMPT = """You are an expert evaluator for a Porto travel assistant. Evaluate the agent's response against the expected response using these criteria:

**Query:** {query}

**Expected Response:** {expected_response}

**Agent Response:** {agent_response}

Rate each dimension on a scale:
- Completeness: [5 = covers all core elements | 3 = partial coverage | 1 = misses core elements]
- Relevance: [5 = stays focused on intent | 3 = minor drift | 1 = off-topic]
- Correctness: [5 = factually accurate | 3 = minor issues | 1 = clear errors]
- Clarity: [5 = concise and readable | 3 = verbose/rough | 1 = hard to parse]
- Structure: [5 = well-organized with headings/lists | 3 = semi-ordered | 1 = unstructured blob]
- Hallucination: [5 = no fabrications | 3 = hints of issues | 1 = clear fabrications]

Respond in this exact JSON format:
{{
  "completeness": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "relevance": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "correctness": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "clarity": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "structure": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "hallucination": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
  "overall_assessment": "<summary>"
}}
"""

class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    reasoning: str

class EvaluationResult(BaseModel):
    completeness: DimensionScore
    relevance: DimensionScore
    correctness: DimensionScore
    clarity: DimensionScore
    structure: DimensionScore
    hallucination: DimensionScore
    overall_assessment: str

llm = AzureChatOpenAI(
    azure_deployment="your-gpt4-deployment",
    openai_api_version="2024-02-01",
    temperature=0
)

structured_llm = llm.with_structured_output(EvaluationResult)

def evaluate_response(query, expected_response, agent_response):
    prompt = EVAL_PROMPT.format(
        query=query,
        expected_response=expected_response,
        agent_response=agent_response
    )
    
    result = structured_llm.invoke(prompt)
    return result.model_dump()