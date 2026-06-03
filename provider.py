from providers.api_client import OpenAIProvider
from providers.local_client import LocalProvider
from logger import logger


class Provider(object):
    def __init__(self, mode: str = "local", model: str = None, **kwargs):
        """
        :param mode: 'local' или 'api' (или 'openai')
        :param model: имя модели (для Ollama например 'llama3.1:8b', для OpenAI 'gpt-4o')
        :param kwargs: дополнительные параметры (api_key, host и т.п.)
        """
        self.mode = mode
        self.model = model
        self.kwargs = kwargs
        self.client = None

        # Сразу создаём нужный объект
        if self.mode == "local":
            logger.info("Инициализация локального провайдера (Ollama)")
            self.client = LocalProvider(model=self.model or "qwen3.5:9b", **kwargs)
        elif self.mode in ("api", "openai"):
            logger.info("Инициализация OpenAI API провайдера")
            self.client = OpenAIProvider(model=self.model or "gpt-4o", **kwargs)
        else:
            raise ValueError(f"Неподдерживаемый режим: {self.mode}. Используйте 'local' или 'api'.")

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:

        logger.info(f"Отправка запроса к провайдеру (mode={self.mode}, model={self.client.model})")
        try:
            response = self.client.complete(system_prompt, user_prompt, **kwargs)
            logger.debug("Ответ успешно получен")
            return response
        except Exception as e:
            logger.exception(f"Ошибка при обращении к провайдеру: {e}")
            raise