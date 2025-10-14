from services.agent import Agent
import os 
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    agent = Agent(model_name="gpt-5-chat")  

    user_query = "Hey, what's your job?"
    print(f"User: {user_query}")
    response = await agent.llm_response(user_query)
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
