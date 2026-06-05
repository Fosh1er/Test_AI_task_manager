from providers.api_client import OpenAIProvider, OpenRouter
from providers.local_client import LocalProvider
from logger import logger
import config


class Provider(object):
    def __init__(self, mode: str = "local", model: str = None, **kwargs):
        """
        :param mode: 'local', 'api', 'openai' или 'openrouter'
        :param model: имя модели
        :param kwargs: дополнительные параметры (api_key, host, num_ctx и т.п.)
        """
        self.mode = mode
        self.model = model
        self.kwargs = kwargs
        self.client = None

        if self.mode == "local":
            logger.info("Инициализация локального провайдера (Ollama)")
            self.client = LocalProvider(model=self.model or "qwen3.5:9b", **kwargs)

        elif self.mode in ("api", "openai"):
            logger.info("Инициализация OpenAI API провайдера")
            api_model = self.model if "gpt" in (self.model or "") else "gpt-4o-mini"
            self.client = OpenAIProvider(
                model=api_model,
                proxy=config.PROXY_URL,
                **kwargs
            )

        elif self.mode == "openrouter":
            logger.info("Инициализация OpenRouter провайдера")
            router_model = self.model or "qwen/qwen3-next-80b-a3b-instruct:free"
            self.client = OpenRouter(
                model=router_model,
                proxy=config.PROXY_URL if hasattr(config, 'PROXY_URL') else None,
                site_url=config.SITE_URL if hasattr(config, 'SITE_URL') else None,
                site_name=config.SITE_NAME if hasattr(config, 'SITE_NAME') else None,
                **kwargs  # передаём все kwargs, OpenRouter сам отфильтрует лишнее
            )

        else:
            raise ValueError(f"Неподдерживаемый режим: {self.mode}. Используйте 'local', 'api' или 'openrouter'.")

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        logger.info(f"Отправка запроса к провайдеру (mode={self.mode}, model={self.client.model})")
        try:
            response = self.client.complete(system_prompt, user_prompt, **kwargs)
            logger.debug("Ответ успешно получен")
            return response
        except Exception as e:
            logger.exception(f"Ошибка при обращении к провайдеру: {e}")
            raise