from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from utils.prompts import llm_judge_prompt
from database.evaluation import EvaluationStorage
import logging
import os
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class JudgeOutput:
    relevance: int  # 0-5 scale
    accuracy: int  # 0-5 scale
    pii_violation: int  # 0 or 1
    safety_violation: int  # 0 or 1
    clarity: int  # 0-5 scale
    overall_score: int  # 0-5 scale
    rationale: str

class EvaluationService(ABC):
    def __init__(self, db_path: str = "./db/evaluation.db"):
        self.db_path = db_path
        self.storage = EvaluationStorage(db_path)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def evaluate(self, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> JudgeOutput:
        """
        Abstract method to evaluate a chatbot response.
        Must be implemented by all subclasses.

        Args:
            user_query: The user's input query
            agent_response: The chatbot's response
            conversation_id: Optional conversation ID for linking
            message_id: Optional message ID for linking

        Returns:
            JudgeOutput: Structured evaluation result
        """
        pass

    def save_evaluation(self, evaluation: JudgeOutput, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> int:
        """Save evaluation result to the database"""
        return self.storage.add_evaluation(
            user_query=user_query,
            assistant_response=agent_response,
            relevance=evaluation.relevance,
            accuracy=evaluation.accuracy,
            pii_violation=evaluation.pii_violation,
            safety_violation=evaluation.safety_violation,
            clarity=evaluation.clarity,
            overall_score=evaluation.overall_score,
            rationale=evaluation.rationale,
            conversation_id=conversation_id,
            message_id=message_id
        )

class LLMJudge(EvaluationService):
    def __init__(self, model_name: str = "gpt-5-chat", db_path: str = "./db/evaluation.db"):
        super().__init__(db_path)
        self.model = AzureChatOpenAI(
            azure_deployment=model_name,
            api_version=os.getenv("OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0,
        )
        self.judge_agent = create_agent(
            model=self.model,
            tools=[],
            response_format=JudgeOutput,
        )

    async def evaluate(self, user_query: str, agent_response: str, conversation_id: Optional[int] = None, message_id: Optional[int] = None) -> JudgeOutput:
        try:
            # Format the judge prompt with the query and response
            formatted_prompt = llm_judge_prompt.format(
                user_query=user_query,
                assistant_response=agent_response
            )
            messages = [SystemMessage(content=llm_judge_prompt)]
            messages.append(HumanMessage(content=formatted_prompt))

            # Invoke the judge agent
            response = await self.judge_agent.ainvoke({"messages": messages})
            judge_output = response.get('structured_response')

            logger.info(f"LLM Judge response: {judge_output}")

            if judge_output is None:
                self.logger.error("LLM Judge returned no response")
                judge_output = self._get_failed_evaluation("No response from judge model")
            else:
                # Ensure pii_violation and safety_violation have default values if None
                if judge_output.pii_violation is None:
                    judge_output.pii_violation = 0
                if judge_output.safety_violation is None:
                    judge_output.safety_violation = 0

            # Save to database
            self.save_evaluation(judge_output, user_query, agent_response, conversation_id, message_id)
            logger.info("Evaluation saved to database")
            logger.info(f"Evaluation details: {judge_output}")
            return judge_output

        except Exception as e:
            self.logger.error(f"LLM Judge evaluation failed: {str(e)}")
            return self._get_failed_evaluation(f"Evaluation error: {str(e)}")

    def _get_failed_evaluation(self, reason: str) -> JudgeOutput:
        """Return a default failed evaluation output"""
        return JudgeOutput(
            relevance=0,
            accuracy=0,
            pii_violation=0,
            safety_violation=0,
            clarity=0,
            overall_score=0,
            rationale=f"Evaluation failed - {reason}"
        )

