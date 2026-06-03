from langchain_core.runnables import RunnableLambda
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import psutil, os

from logger import logger
from provider import Provider
from adapter import create_langchain_adapter
from prompt_loader import load_prompt
from text_chunker import chunk_text

def extract_json_from_text(text: str) -> str:
    """
    Извлекает JSON-массив из текста, удаляя лишний шум,
    Markdown-разметку и пояснения LLM.
    """
    # Поиск первой открывающей и последней закрывающей квадратной скобки
    start = text.find('[')
    end = text.rfind(']')

    if start == -1 or end == -1 or end < start:
        return text # Возвращаем как есть, если массив не найден (парсер сам выдаст ошибку)

    return text[start:end+1]

def build_extraction_chain(llm):
    system = load_prompt("system_extract.md")
    user_tmpl = load_prompt("user_extract.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_tmpl),
    ])
    # Возвращаем цепочку БЕЗ парсера, чтобы иметь доступ к сырому тексту
    return prompt | llm | RunnableLambda(extract_json_from_text)


def build_aggregation_chain(llm):
    system = load_prompt("system_aggregate.md")
    user_tmpl = load_prompt("user_aggregate.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_tmpl),
    ])
    return prompt | llm | JsonOutputParser()


def run_pipeline(
        transcript: str,
        meeting_date: str,
        provider_mode: str = "local",
        model: str = "qwen3.5:2b",
        chunk_size: int = 6000,
        overlap: int = 500,
        llm_kwargs: dict = None
    ) -> list[dict]:
    """
    Полный цикл: чанкование -> извлечение из чанков -> агрегация.
    """
    logger.info(f"=== Запуск пайплайна для даты {meeting_date} ===")

    if llm_kwargs is None:
        llm_kwargs = {}

    provider = Provider(mode=provider_mode, model=model, **llm_kwargs)
    llm = create_langchain_adapter(provider)

    extract_chain = build_extraction_chain(llm)
    aggregate_chain = build_aggregation_chain(llm)

    chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        logger.warning("Чанки отсутствуют, возвращается пустой список")
        return []

    all_partial = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Обработка чанка {idx}/{len(chunks)}...")
        try:
            # Теперь используем extract_chain, которая возвращает строку (благодаря удалению JsonOutputParser)
            raw_output = extract_chain.invoke({"chunk_text": chunk, "meeting_date": meeting_date})

            # Теперь пробуем распарсить вручную
            tasks = JsonOutputParser().parse(raw_output)
            all_partial.append(tasks)
        except Exception as e:
            logger.error(f"Ошибка при обработке чанка {idx}: {e}")
            if 'raw_output' in locals():
                logger.error(f"RAW OUTPUT FROM LLM: {raw_output}")
            all_partial.append([])

    logger.info("Запуск агрегации частичных списков...")
    partial_json_str = json.dumps(all_partial, ensure_ascii=False, indent=2)
    try:
        final_tasks = aggregate_chain.invoke({"partial_jsons": partial_json_str})
        logger.info(f"Агрегация завершена, итоговых задач: {len(final_tasks)}")
    except Exception as e:
        logger.exception("Ошибка при агрегации")
        final_tasks = []

    return final_tasks
