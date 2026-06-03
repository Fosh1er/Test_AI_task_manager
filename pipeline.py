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
    # Добавляем шаг очистки текста перед парсингом JSON
    return prompt | llm | RunnableLambda(extract_json_from_text) | JsonOutputParser()


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
        provider_mode: str = "local",
        model: str = "qwen3.5:9b",
        chunk_size: int = 6000,
        overlap: int = 500,
        llm_kwargs: dict = None  # <-- новый параметр
) -> list[dict]:
    """
    Полный цикл: чанкование -> извлечение из чанков -> агрегация.

    :param llm_kwargs: дополнительные параметры для провайдера (num_ctx, num_thread, num_gpu и т.д.)
    """
    logger.info("=== Запуск пайплайна обработки транскрипта ===")
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024 ** 2
    logger.info(f"Память перед созданием провайдера: {mem_before:.1f} МБ")

    # Параметры по умолчанию
    if llm_kwargs is None:
        llm_kwargs = {}

    # 1. Создаём провайдер с переданными ограничениями
    provider = Provider(mode=provider_mode, model=model, **llm_kwargs)

    mem_after_provider = proc.memory_info().rss / 1024 ** 2
    logger.info(f"Память после создания провайдера: {mem_after_provider:.1f} МБ")

    llm = create_langchain_adapter(provider)

    mem_after_lang_adapter = proc.memory_info().rss / 1024 ** 2
    logger.info(f"Память после создания адаптера: {mem_after_lang_adapter:.1f} МБ")

    # 2. Цепочки
    extract_chain = build_extraction_chain(llm)
    aggregate_chain = build_aggregation_chain(llm)

    mem_after_lang_chain = proc.memory_info().rss / 1024 ** 2
    logger.info(f"Память после создания цепи: {mem_after_lang_chain:.1f} МБ")

    # 3. Чанкование
    chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        logger.warning("Чанки отсутствуют, возвращается пустой список")
        return []

    # 4. Обработка каждого чанка
    all_partial = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Обработка чанка {idx}/{len(chunks)}...")
        try:
            tasks = extract_chain.invoke({"chunk_text": chunk})
            logger.debug(f"Из чанка {idx} извлечено задач: {len(tasks)}")
            all_partial.append(tasks)
        except Exception as e:
            logger.error(f"Ошибка при обработке чанка {idx}: {e}")
            all_partial.append([])

    # 5. Агрегация
    logger.info("Запуск агрегации частичных списков...")
    partial_json_str = json.dumps(all_partial, ensure_ascii=False, indent=2)
    try:
        final_tasks = aggregate_chain.invoke({"partial_jsons": partial_json_str})
        logger.info(f"Агрегация завершена, итоговых задач: {len(final_tasks)}")
    except Exception as e:
        logger.exception("Ошибка при агрегации")
        final_tasks = []

    logger.info("=== Пайплайн завершён ===")
    return final_tasks