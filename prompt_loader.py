# prompt_loader.py
import os
import re
from logger import logger

PROMPTS_DIR = "prompts"

def load_prompt(filename: str) -> str:
    """Читает .md файл, извлекает текст после заголовка '## Prompt'."""
    filepath = os.path.join(PROMPTS_DIR, filename)
    logger.debug(f"Загрузка промпта из {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        logger.error(f"Файл промпта не найден: {filepath}")
        raise

    # Ищем блок, начинающийся с "## Prompt" и забираем всё до конца файла
    match = re.search(r'## Prompt\s*\n(.*)', content, re.DOTALL)
    if not match:
        logger.error(f"Блок '## Prompt' не найден в {filename}")
        raise ValueError(f"Формат файла нарушен: {filename}")

    prompt_text = match.group(1).strip()
    logger.debug(f"Промпт загружен, длина {len(prompt_text)} символов")
    return prompt_text