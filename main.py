import os
import asyncio
import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic import BaseModel

load_dotenv()

# 1. Определяем модель выхода
class CalcOutput(BaseModel):
    result: float
    explanation: str

# 2. Создаём агента с нужными параметрами
agent = Agent(
    model="openai:gpt-3.5-turbo",
    system_prompt=(
        "Ты — калькулятор, который получает арифметические выражения и возвращает JSON с полями "
        "'result' (число) и 'explanation' (строка с объяснением)."
    ),
    output_type=CalcOutput
)

logfire.configure(
    token=os.getenv("LOGFIRE_KEY")
)
logfire.instrument_pydantic_ai()

async def main():
    prompt = "2 + 2"
    result = await agent.run(prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())