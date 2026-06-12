from langchain_core.runnables import RunnableLambda
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from logger import logger
from provider import Provider
from adapter import create_langchain_adapter
from prompt_loader import load_prompt
from text_chunker import chunk_text
from date_normalizer import normalize_tasks_dates


def extract_json_from_text(text: str) -> str:
    """Извлекает JSON-массив из текста, удаляя шум и markdown."""
    if not text:
        return "[]"
    # Убираем markdown-обёртку типа ```json ... ```
    text = text.strip()
    if text.startswith("```"):
        # Снимаем первую и последнюю строку
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1 or end < start:
        return "[]"
    return text[start:end + 1]


def build_extraction_chain(llm):
    system = load_prompt("system_extract.md")
    user_tmpl = load_prompt("user_extract.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_tmpl),
    ])
    return prompt | llm | RunnableLambda(extract_json_from_text)


def build_aggregation_chain(llm):
    system = load_prompt("system_aggregate.md")
    user_tmpl = load_prompt("user_aggregate.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_tmpl),
    ])
    return prompt | llm | RunnableLambda(extract_json_from_text)


def build_validation_chain(llm):
    system = load_prompt("system_validate.md")
    user_tmpl = load_prompt("user_validate.md")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user_tmpl),
    ])
    return prompt | llm | RunnableLambda(extract_json_from_text)


def _safe_parse_json(raw: str) -> list | dict:
    """Безопасный парсинг JSON."""
    try:
        return JsonOutputParser().parse(raw)
    except Exception as e:
        logger.warning(f"JSON parse error: {e}")
        logger.debug(f"Raw: {raw[:500]}")
        return []


def run_pipeline(
        transcript: str,
        meeting_date: str,
        provider_mode: str = "local",
        model: str = "qwen3.5:2b",
        chunk_size: int = 6000,
        overlap: int = 500,
        llm_kwargs: dict = None,
        enable_validation: bool = True,
) -> list[dict]:
    """
    Полный цикл:
    1. Чанкование
    2. Извлечение задач из каждого чанка
    3. Агрегация (дедупликация)
    4. Нормализация дат алгоритмически
    5. Валидация через LLM-судью (опционально)
    """
    logger.info(f"=== Запуск пайплайна для даты {meeting_date} ===")

    if llm_kwargs is None:
        llm_kwargs = {}

    provider = Provider(mode=provider_mode, model=model, **llm_kwargs)
    llm = create_langchain_adapter(provider)

    extract_chain = build_extraction_chain(llm)
    aggregate_chain = build_aggregation_chain(llm)
    validate_chain = build_validation_chain(llm) if enable_validation else None

    chunks = chunk_text(transcript, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        logger.warning("Чанки отсутствуют")
        return []

    # === Шаг 1: извлечение из каждого чанка ===
    all_partial = []
    for idx, chunk in enumerate(chunks, 1):
        logger.info(f"Обработка чанка {idx}/{len(chunks)}...")
        try:
            raw_output = extract_chain.invoke({
                "chunk_text": chunk,
                "meeting_date": meeting_date,
            })
            tasks = _safe_parse_json(raw_output)
            if not isinstance(tasks, list):
                tasks = []
            all_partial.append(tasks)
            logger.info(f"  Чанок {idx}: извлечено задач: {len(tasks)}")
        except Exception as e:
            logger.error(f"Ошибка при обработке чанка {idx}: {e}")
            all_partial.append([])

    # === Шаг 2: агрегация ===
    logger.info("Запуск агрегации...")
    partial_json_str = json.dumps(all_partial, ensure_ascii=False, indent=2)
    try:
        raw_agg = aggregate_chain.invoke({
            "partial_jsons": partial_json_str,
            "meeting_date": meeting_date,
        })
        final_tasks = _safe_parse_json(raw_agg)
        if not isinstance(final_tasks, list):
            final_tasks = []
        logger.info(f"Агрегация завершена, задач: {len(final_tasks)}")
    except Exception as e:
        logger.exception("Ошибка при агрегации")
        final_tasks = []

    # === Шаг 3: нормализация дат алгоритмически ===
    logger.info("Нормализация дат...")
    final_tasks = normalize_tasks_dates(final_tasks, meeting_date)

    # === Шаг 4: валидация через LLM-судью ===
    if enable_validation and validate_chain and final_tasks:
        logger.info("Запуск валидации...")
        try:
            tasks_json_str = json.dumps(final_tasks, ensure_ascii=False, indent=2)
            raw_val = validate_chain.invoke({
                "transcript": transcript,
                "tasks_json": tasks_json_str,
                "meeting_date": meeting_date,
            })
            validation = _safe_parse_json(raw_val)
            if isinstance(validation, dict):
                final_tasks = _apply_validation(final_tasks, validation)
                logger.info(f"После валидации задач: {len(final_tasks)}")
            else:
                logger.warning("Валидация вернула не dict, пропускаем")
        except Exception as e:
            logger.warning(f"Ошибка валидации (не критично): {e}")

    return final_tasks


def _apply_validation(tasks: list[dict], validation: dict) -> list[dict]:
    """
    Применяет результат валидации:
    - убирает задачи с ошибками (если есть исправление — заменяет)
    - добавляет пропущенные задачи
    """
    valid_indices = set(validation.get("валидные_задачи", []))
    errors = {e["индекс"]: e for e in validation.get("ошибки", []) if "индекс" in e}
    missed = validation.get("пропущенные_задачи", [])

    result = []
    for i, task in enumerate(tasks):
        if i in valid_indices:
            result.append(task)
        elif i in errors:
            err = errors[i]
            if err.get("исправление"):
                result.append(err["исправление"])
            # иначе — пропускаем задачу (ошибка не исправлена)

    # Добавляем пропущенные
    for m in missed:
        if isinstance(m, dict) and "задача" in m:
            result.append(m)

    return result