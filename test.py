# test.py
import ollama
from logger import logger
import config

def main():
    model_name = config.MODEL_NAME
    kwargs = config.LLM_KWARGS
    logger.info(f"Тест модели {model_name} с параметрами {kwargs}")

    client = ollama.Client()

    options = {
        "num_gpu": kwargs.get("num_gpu", -1),
        "num_ctx": kwargs.get("num_ctx", 4096),
        "num_thread": kwargs.get("num_thread", 4),
    }
    logger.info(f"Отправка запроса с options={options}")
    try:
        resp = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": "Ты тестовый ассистент."},
                {"role": "user", "content": "Скажи: Привет, я работаю"}
            ],
            options=options,
            stream=False
        )
        logger.info("Ответ получен. Вывожу атрибуты ChatResponse:")
        logger.info(f"  model: {resp.model}")
        logger.info(f"  message: {resp.message}")
        logger.info(f"  done: {resp.done}")
        logger.info(f"  total_duration: {resp.total_duration}")
        logger.info(f"  load_duration: {resp.load_duration}")
        logger.info(f"  prompt_eval_count: {resp.prompt_eval_count}")
        logger.info(f"  prompt_eval_duration: {resp.prompt_eval_duration}")
        logger.info(f"  eval_count: {resp.eval_count}")
        logger.info(f"  eval_duration: {resp.eval_duration}")
        # Детали с GPU-информацией
        if hasattr(resp, 'details'):
            logger.info(f"  details: {resp.details}")
        # Полный словарь ответа
        try:
            resp_dict = resp.model_dump()
            logger.info("Полное содержимое ответа (model_dump):")
            for k, v in resp_dict.items():
                logger.info(f"  {k}: {v}")
        except Exception as e:
            logger.warning(f"Не удалось преобразовать ответ в словарь: {e}")
    except Exception as e:
        logger.exception("Ошибка прямого вызова")

if __name__ == "__main__":
    main()