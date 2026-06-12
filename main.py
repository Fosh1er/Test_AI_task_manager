import sys
import os

# 1. Очистка переменных окружения
for key in list(os.environ.keys()):
    if key.lower() in ['http_proxy', 'https_proxy', 'all_proxy']:
        del os.environ[key]

# 2. Патч httpx
try:
    import httpx
    def dummy_get_proxy_map(self, proxy, allow_env_proxies):
        return {}
    httpx.Client._get_proxy_map = dummy_get_proxy_map
    print("SOCKS4 protection: httpx proxy map disabled successfully.")
except Exception as e:
    print(f"SOCKS4 protection failed to apply: {e}")

import pandas as pd
import sqlite3
from pathlib import Path
from logger import logger
from pipeline import run_pipeline
import config

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
    llm_kwargs = config.LLM_KWARGS

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
            llm_kwargs=llm_kwargs,
            enable_validation=True,  # включаем валидацию
        )

        df = pd.DataFrame(tasks)
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

    # === Отчёт о стабильности ===
    print("\n" + "=" * 60)
    print(f"ОТЧЕТ ПО СТАБИЛЬНОСТИ: {filename}")
    print("=" * 60)
    all_same = True
    first_stat = stats[0]
    diff_runs = []

    for s in stats:
        print(f"run_{s['run']}: Выполненные={s['completed']}, "
              f"Невыполненные={s['failed']}, Новые={s['new']}, Всего={s['total']}")
        if s != first_stat:
            all_same = False
            diff_runs.append(f"run_{s['run']}")

    print(f"\nСтабильность по количеству: {'Да' if all_same else 'Нет'}")
    if not all_same:
        print(f"Отличаются прогоны: {', '.join(diff_runs)}")

    # === Итоговый датасет (прогон 1) ===
    print("\nИТОГОВЫЙ ДАТАСЕТ (Прогон 1):")
    df_final = results[0]
    if not df_final.empty:
        print(df_final.to_string())
    else:
        print("Задачи не найдены.")

    # === Сохранение в CSV (просьба руководителя) ===
    base_name = Path(filename).stem
    csv_path = Path(f"{base_name}_tasks.csv")
    if not df_final.empty:
        df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ CSV сохранён: {csv_path}")

    # === Задача со звёздочкой: SQLite ===
    sqlite_path = Path("result.sqlite")
    if not df_final.empty:
        try:
            conn = sqlite3.connect(sqlite_path)
            df_final.to_sql(base_name, conn, if_exists="replace", index=False)
            conn.close()
            print(f"✓ SQLite сохранён: {sqlite_path} (таблица '{base_name}')")
        except Exception as e:
            logger.warning(f"Не удалось сохранить в SQLite: {e}")

    print("\n" + "=" * 60 + "\n")


def main():
    test_files = {
        "transcript.txt": "2026-04-13",
        "transcript2.txt": "2026-04-29",
        "transcript3.txt": "2026-04-15",
    }

    for filename, date in test_files.items():
        try:
            run_stability_test(filename, date)
        except Exception as e:
            logger.exception(f"Ошибка при обработке файла {filename}: {e}")


if __name__ == "__main__":
    main()