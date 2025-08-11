import os
import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from simpleeval import simple_eval

# получаем ключи API из переменных окружения
load_dotenv()

# подключаемся к Logfire для логирования и мониторинга
logfire.configure(token=os.getenv("LOGFIRE_KEY"))
logfire.instrument_pydantic_ai()

# описывем модель ответа
class ModelOutput(BaseModel):
    result: float = Field(description="Результат вычислений")
    explanation: str = Field(description="Результаты проверки и пояснение")

# инициализируем агента
agent = Agent(
    model='openai:gpt-3.5-turbo',
    system_prompt=(
        "Ты — калькулятор, который объясняет свои действия. "
        "Прежде чем дать финальный ответ, ты должен сгенерировать простой тест (например, 2+2=4), "
        "вызвать свой инструмент `calculate` для проверки результата. "
        "Если результат инструмента не совпал с ожидаемым, ты должен повторить попытку. "
        "После успешной проверки ты даешь ответ в формате Pydantic-модели."
    ),
    output_type=ModelOutput, # подключаем модель ответа
    retries=2, # настраиваеаем количество ретраев
    instrument=True
)

# инициализируем инструменты агента
@agent.tool
def calculate(ctx, expression: str) -> float:
    """Вычисляет значение математического выражения."""
    try:
        return float(simple_eval(expression)) # используем simple_eval для безопасного вычисления
    except Exception as e:
        raise ValueError(f"Ошибка вычисления: {e}")

response = agent.run_sync("Сколько будет 418*2/4?")
print(f"Результат: {response.output.result}")
print(f"Пояснение: {response.output.explanation}")