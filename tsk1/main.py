import os
import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from simpleeval import simple_eval

load_dotenv()

logfire.configure(token=os.getenv("LOGFIRE_KEY"))
logfire.instrument_pydantic_ai()

class ModelOutput(BaseModel):
    result: float = Field(description="Результат вычислений")
    explanation: str = Field(description="Результаты проверки и пояснение")

agent = Agent(
    model='openai:gpt-3.5-turbo',
    system_prompt=(
        "Ты — калькулятор, который объясняет свои действия. "
        "Прежде чем дать финальный ответ, ты должен сгенерировать простой тест (например, 2+2=4), "
        "вызвать свой инструмент `calculate` для проверки результата. "
        "Если результат инструмента не совпал с ожидаемым, ты должен повторить попытку. "
        "После успешной проверки ты даешь ответ в формате Pydantic-модели."
    ),
    output_type=ModelOutput,
    instrument=True
)

@agent.tool
def calculate(ctx, expression: str) -> float:
    """Вычисляет значение математического выражения."""
    try:
        return float(simple_eval(expression))
    except Exception as e:
        raise ValueError(f"Ошибка вычисления: {e}")

response = agent.run_sync("Сколько будет 3*3+1?")
print(f"Результат: {response.output.result}")
print(f"Пояснение: {response.output.explanation}")