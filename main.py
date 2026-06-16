import sys
import os
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv

# === НАДЕЖНАЯ ЗАГРУЗКА .ENV ===
ENV_PATH = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=ENV_PATH, override=True)

# 1. Очистка переменных окружения (для корректной работы прокси/сети)
for key in list(os.environ.keys()):
    if key.lower() in ['http_proxy', 'https_proxy', 'all_proxy']:
        del os.environ[key]

# 2. Патч httpx (защита от ошибок SOCKS)
try:
    import httpx


    def dummy_get_proxy_map(self, proxy, allow_env_proxies):
        return {}


    httpx.Client._get_proxy_map = dummy_get_proxy_map
except Exception:
    pass

import pandas as pd
import requests
from logger import logger
from pipeline import run_pipeline
import config

os.environ['PYTHONUNBUFFERED'] = '1'


# ==========================================
# БЛОК РАБОТЫ С ЯНДЕКС ДИСКОМ
# ==========================================
def list_public_yandex_disk_files(public_key: str, path: str = "/") -> list:
    url = "https://cloud-api.yandex.net/v1/disk/public/resources"
    params = {"public_key": public_key, "path": path, "limit": 100}
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get("_embedded", {}).get("items", [])
        return [{"name": i["name"], "type": i["type"], "path": i.get("path", i["name"])} for i in items]
    except Exception as e:
        logger.error(f"Ошибка при получении списка файлов: {e}")
        return []


def load_transcript_from_yandex_disk(filename: str) -> str:
    token = os.environ.get("YANDEX_DISK_TOKEN")
    public_key = os.environ.get("YANDEX_DISK_PUBLIC_KEY")
    disk_path = os.environ.get("YANDEX_DISK_PATH", "").strip("/")

    if token:
        full_path = f"/{disk_path}/{filename}" if disk_path else f"/{filename}"
        encoded_path = urllib.parse.quote(full_path, safe="/")
        url = f"https://cloud-api.yandex.net/v1/disk/resources/download?path={encoded_path}"
        response = requests.get(url, headers={"Authorization": f"OAuth {token}"})
    elif public_key:
        info_resp = requests.get("https://cloud-api.yandex.net/v1/disk/public/resources",
                                 params={"public_key": public_key})
        info_resp.raise_for_status()
        res_type = info_resp.json().get('type')
        file_path = "" if res_type == 'file' else filename
        url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
        params = {"public_key": public_key}
        if file_path: params["path"] = file_path
        response = requests.get(url, params=params)
    else:
        raise ValueError("Не задан ни YANDEX_DISK_TOKEN, ни YANDEX_DISK_PUBLIC_KEY")

    if response.status_code != 200:
        logger.error(f"Ошибка API: {response.status_code} | {response.text}")
        response.raise_for_status()

    download_url = response.json().get("href")
    file_response = requests.get(download_url)
    file_response.raise_for_status()
    return file_response.content.decode("utf-8")


def load_transcript(filename: str) -> str:
    source = os.environ.get("TRANSCRIPT_SOURCE", "local").lower()
    if source == "yandex_disk":
        return load_transcript_from_yandex_disk(filename)
    else:
        filepath = Path(config.TRANSCRIPTS_DIR) / filename
        if not filepath.exists(): raise FileNotFoundError(filepath)
        return filepath.read_text(encoding="utf-8")


# ==========================================
# БЛОК ТЕСТИРОВАНИЯ СТАБИЛЬНОСТИ
# ==========================================
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    return (df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
            .sort_values(by=list(df.columns))
            .reset_index(drop=True))


def dfs_are_identical(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Строгое сравнение двух датафреймов на полную идентичность."""
    if df1.empty and df2.empty: return True
    if df1.empty or df2.empty: return False

    # Сортируем и сбрасываем индексы для корректного сравнения
    d1 = normalize_df(df1).fillna("").to_dict(orient="records")
    d2 = normalize_df(df2).fillna("").to_dict(orient="records")

    # Сортируем списки словарей, чтобы порядок строк не влиял на сравнение
    try:
        return sorted(d1, key=lambda x: str(x)) == sorted(d2, key=lambda x: str(x))
    except Exception:
        return False


def run_stability_test(filename: str, meeting_date: str):
    logger.info("=" * 60)
    logger.info(f"Тестирование файла: {filename} (Дата встречи: {meeting_date})")

    transcript = load_transcript(filename)

    # ЖЕСТКИЕ ПАРАМЕТРЫ ДЛЯ ДЕТЕРМИНИЗМА (ТЗ п.4)
    # temperature=0.0 отключает случайность.
    # seed=42 фиксирует зерно генерации псевдослучайных чисел в модели.
    llm_kwargs = {
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 4096
    }

    results = []
    stats = []

    # Ровно 5 прогонов (ТЗ п.2)
    for run in range(1, 6):
        logger.info(f"--- Прогон {run}/5 ---")
        tasks = run_pipeline(
            transcript, meeting_date=meeting_date,
            provider_mode=config.PROVIDER_MODE, model=config.MODEL_NAME,
            chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP,
            llm_kwargs=llm_kwargs, enable_validation=True,
        )

        df = pd.DataFrame(tasks)
        if not df.empty:
            # Удаляем служебные поля, если они есть
            for col in ['анализ', 'исходный_срок']:
                if col in df.columns: df = df.drop(columns=[col])

            df = df.rename(columns={
                "блок": "Блок", "задача": "Задача", "ответственный": "Ответственный",
                "срок": "Срок", "обоснование": "Обоснование"
            })

        results.append(df)

        if not df.empty:
            counts = df['Блок'].value_counts()
            stats.append({
                "run": run, "completed": counts.get("Выполненные", 0),
                "failed": counts.get("Невыполненные", 0), "new": counts.get("Новые", 0), "total": len(df)
            })
        else:
            stats.append({"run": run, "completed": 0, "failed": 0, "new": 0, "total": 0})

    # === АНАЛИЗ РЕЗУЛЬТАТОВ ===
    print("\n" + "=" * 60)
    print(f"ОТЧЕТ ПО СТАБИЛЬНОСТИ: {filename}")
    print("=" * 60)

    for s in stats:
        print(
            f"Прогон {s['run']}: Выполненные={s['completed']}, Невыполненные={s['failed']}, Новые={s['new']}, Всего={s['total']}")

    # Строгая проверка на идентичность всех 5 прогонов
    base_df = results[0]
    all_identical = True
    diff_runs = []

    for i in range(1, 5):
        if not dfs_are_identical(base_df, results[i]):
            all_identical = False
            diff_runs.append(f"Прогон {i + 1}")

    print("-" * 60)
    if all_identical:
        print("✅ ВЕРДИКТ: СТАБИЛЬНОСТЬ 100% (Все 5 прогонов абсолютно идентичны)")
    else:
        print(f"❌ ВЕРДИКТ: НЕСТАБИЛЬНО (Отличаются: {', '.join(diff_runs)})")
        print("Примечание: Если вы используете бесплатные модели OpenRouter (:free),")
        print("они могут игнорировать параметр seed и выдавать плавающий результат.")
        print("Для 100% детерминизма требуется платная модель или прямой API OpenAI.")
    print("-" * 60)

    # === ИТОГОВЫЙ ДАТАСЕТ ===
    print("\nИТОГОВЫЙ ДАТАСЕТ (Результат Прогон 1):")
    if not base_df.empty:
        print(normalize_df(base_df).to_string())
    else:
        print("Задачи не найдены.")

    print("\n" + "=" * 60 + "\n")


def main():
    # Запуск на всех трех файлах из задания (ТЗ п.3)
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