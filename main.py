import sys
import pandas as pd
from pathlib import Path
from logger import logger
from pipeline import run_pipeline
import config  # наш конфиг

def load_transcript(filename: str) -> str:
    filepath = Path(config.TRANSCRIPTS_DIR) / filename
    if not filepath.exists():
        logger.error(f"Файл не найден: {filepath}")
        raise FileNotFoundError(filepath)
    logger.info(f"Загрузка транскрипта: {filepath}")
    return filepath.read_text(encoding="utf-8")

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    return (df
            .map(lambda x: str(x).strip() if isinstance(x, str) else x)
            .sort_values(by=list(df.columns))
            .reset_index(drop=True))

def main(filename: str = "transcript.txt"):
    logger.info("=" * 50)
    logger.info("Запуск проверки стабильности извлечения задач")
    logger.info(f"Провайдер: {config.PROVIDER_MODE}, модель: {config.MODEL_NAME}")

    transcript = load_transcript(filename)

    # Для локального провайдера передаём LLM_KWARGS, для API — пустой словарь
    llm_kwargs = config.LLM_KWARGS if config.PROVIDER_MODE == "local" else {}

    results = []
    for run in range(1, 6):
        logger.info(f"----- Прогон {run}/5 -----")
        tasks = run_pipeline(
            transcript,
            provider_mode=config.PROVIDER_MODE,
            model=config.MODEL_NAME,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            llm_kwargs=llm_kwargs
        )
        df = pd.DataFrame(tasks)
        if df.empty:
            logger.warning(f"Прогон {run} вернул пустой список задач")
        else:
            logger.info(f"Прогон {run}: извлечено задач {len(df)}")
        results.append(normalize_df(df))

    base = results[0]
    stable = True
    for i, df in enumerate(results[1:], 2):
        if not base.equals(df):
            stable = False
            logger.warning(f"Различие между прогоном 1 и {i}")
    logger.info(f"Результат: {'СТАБИЛЬНО' if stable else 'НЕ СТАБИЛЬНО'}")
    print(f"\nСтабильность за 5 прогонов: {'STABLE' if stable else 'NOT STABLE'}")

    base.to_csv("tasks_extracted.csv", index=False, encoding="utf-8")
    logger.info("Первый прогон сохранён в tasks_extracted.csv")

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "transcript.txt"
    main(filename=fname)