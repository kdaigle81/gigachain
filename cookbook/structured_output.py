from __future__ import annotations

from typing import List

from dotenv import find_dotenv, load_dotenv
from gigachat import GigaChat as GigaChatSDK
from langchain_gigachat import GigaChat
from pydantic import BaseModel, Field


load_dotenv(find_dotenv())

PROMPT = "Реши уравнение 8x + 7 = -23 пошагово."


class MathAnswer(BaseModel):
    """Structured answer for a math task."""

    steps: List[str] = Field(description="Пошаговое решение")
    final_answer: str = Field(description="Итоговый ответ")


def print_answer(title: str, answer: MathAnswer) -> None:
    print(f"\n=== {title} ===")
    print("Шаги:")
    for step in answer.steps:
        print(f"- {step}")
    print("Ответ:", answer.final_answer)


def sdk_chat_parse_example() -> None:
    """Use native gigachat SDK structured output helper."""
    with GigaChatSDK() as client:
        completion, parsed = client.chat_parse(
            PROMPT,
            response_format=MathAnswer,
            strict=True,
        )

    print_answer("gigachat SDK: chat_parse()", parsed)
    print("Токенов использовано:", completion.usage.total_tokens)


def langchain_function_calling_example() -> None:
    """Use langchain-gigachat default structured output mode."""
    llm = GigaChat(model="GigaChat-2-Max", top_p=0)
    structured_llm = llm.with_structured_output(MathAnswer)

    parsed = structured_llm.invoke(PROMPT)
    print_answer("langchain-gigachat: default function_calling", parsed)


def langchain_json_schema_example() -> None:
    """Use langchain-gigachat native JSON Schema response_format mode."""
    llm = GigaChat(
        model="GigaChat-2-Max",
        top_p=0,
        function_ranker={"enabled": False},
    )
    structured_llm = llm.with_structured_output(MathAnswer, method="json_schema")

    parsed = structured_llm.invoke(PROMPT)
    print_answer('langchain-gigachat: method="json_schema"', parsed)


if __name__ == "__main__":
    sdk_chat_parse_example()
    langchain_function_calling_example()
    langchain_json_schema_example()
