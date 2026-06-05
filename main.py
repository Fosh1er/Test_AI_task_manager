import sys
import os

# 1. Очистка переменных окружения (на всякий случай)
for key in list(os.environ.keys()):
    if key.lower() in ['http_proxy', 'https_proxy', 'all_proxy']:
        del os.environ[key]

# 2. Ядерный патч httpx: заставляем библиотеку игнорировать любые системные прокси
try:
    import httpx
    def dummy_get_proxy_map(self, proxy, allow_env_proxies):
        return {}
    # Перезаписываем метод поиска прокси в самом классе Client
    httpx.Client._get_proxy_map = dummy_get_proxy_map
    print("SOCKS4 protection: httpx proxy map disabled successfully.")
except Exception as e:
    print(f"SOCKS4 protection failed to apply: {e}")

import pandas as pd
from pathlib import Path
from logger import logger
from pipeline import run_pipeline
import config

# Принудительный сброс буфера вывода для PyCharm/IDE
os.environ['PYTHONUNBUFFERED'] = '1'

def load_transcript(filename: str) -> str:
    filepath = Path(config.TRANSCRIPTS_DIR) / filename
    if not filepath.exists():
        logger.error(f"Файл не найден: {filepath}")
        raise FileNotFoundError(filepath)
    return filepath.read_text(encoding="utf-8")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (df
            .map(lambda x: str(x).strip() if isinstance(x, str) else x)
            .sort_values(by=list(df.columns))
            .reset_index(drop=True))

def run_stability_test(filename: str, meeting_date: str):
    logger.info("=" * 60)
    logger.info(f"Тестирование файла: {filename} (Дата встречи: {meeting_date})")

    transcript = load_transcript(filename)
    llm_kwargs = config.LLM_KWARGS if config.PROVIDER_MODE == "local" else {"api_key": config.OPENAI_API_KEY}

    results = []
    stats = []

    for run in range(1, 6):
        logger.info(f"--- Прогон {run}/5 ---")
        tasks = run_pipeline(
            transcript,
            meeting_date=meeting_date,
            provider_mode=config.PROVIDER_MODE,
            model=config.MODEL_NAME,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            llm_kwargs=llm_kwargs
        )

        df = pd.DataFrame(tasks)
        # Приводим к нужному формату колонок согласно ТЗ
        if not df.empty:
            df = df.rename(columns={
                "блок": "Блок",
                "задача": "Задача",
                "ответственный": "Ответственный",
                "срок": "Срок",
                "обоснование": "Обоснование"
            })
            df = normalize_df(df)

        results.append(df)

        # Считаем статистику для отчета
        if not df.empty:
            counts = df['Блок'].value_counts()
            completed = counts.get("Выполненные", 0)
            failed = counts.get("Невыполненные", 0)
            new = counts.get("Новые", 0)
            total = len(df)
        else:
            completed = failed = new = total = 0

        stats.append({
            "run": run,
            "completed": completed,
            "failed": failed,
            "new": new,
            "total": total
        })

    # Вывод отчета о стабильности
    print("\nОТЧЕТ ПО СТАБИЛЬНОСТИ:")
    all_same = True
    first_stat = stats[0]
    diff_runs = []

    for s in stats:
        print(f"run_{s['run']}: Выполненные={s['completed']}, Невыполненные={s['failed']}, Новые={s['new']}, Всего={s['total']}")
        if s != first_stat:
            all_same = False
            diff_runs.append(f"run_{s['run']}")

    print(f"Стабильность по количеству: {'Да' if all_same else 'Нет'}")
    if not all_same:
        print(f"Отличаются прогоны: {', '.join(diff_runs)}")

    # Вывод итогового датасета (первого прогона)
    print("\nИТОГОВЫЙ ДАТАСЕТ (Прогон 1):")
    if not results[0].empty:
        print(results[0].to_string())
    else:
        print("Задачи не найдены.")
    print("\n" + "=" * 60 + "\n")

def main():
    # Соответствие файлов и дат встречи из ТЗ
    test_files = {
        #"transcript.txt": "2026-04-13",
        #"transcript2.txt": "2026-04-29",
        #"transcript3.txt": "2026-04-15",
        "transcript_discord.txt": "2026-06-05"
    }

    for filename, date in test_files.items():
        try:
            run_stability_test(filename, date)
        except Exception as e:
            logger.exception(f"Ошибка при обработке файла {filename}: {e}")

if __name__ == "__main__":
    main()
