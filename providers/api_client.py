from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import time
import os
import httpx
from dotenv import load_dotenv
from logger import logger

load_dotenv()


class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, proxy: str = None, **kwargs):
        self.model = model

        client_args = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}

        if proxy:
            logger.info(f"OpenAI: Использование прокси: {proxy}")
            client_args["http_client"] = httpx.Client(proxy=proxy)

        self.client = OpenAI(**client_args)
        self.last_response_info = {}

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=kwargs.get("temperature", 0.01),
            max_tokens=kwargs.get("max_tokens", 1024),
            seed=kwargs.get("seed")
        )
        elapsed = time.time() - start_time
        content = response.choices[0].message.content
        usage = response.usage
        tokens = usage.completion_tokens if usage else None
        tokens_per_sec = tokens / elapsed if tokens else None

        self.last_response_info = {
            "tokens": tokens,
            "time_seconds": elapsed,
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "total_tokens": usage.total_tokens if usage else None
        }
        logger.debug(f"OpenAI: {tokens} tokens in {elapsed:.2f}s "
                     f"({tokens_per_sec:.1f} tok/s)" if tokens_per_sec else "")
        return content


class OpenRouter:
    # Параметры retry
    MAX_RETRIES = 5
    BASE_DELAY = 5       # базовая задержка в секундах
    MAX_DELAY = 120      # максимум между попытками

    def __init__(self, model: str = "qwen/qwen3-next-80b-a3b-instruct:free",
                 api_key: str = None, proxy: str = None,
                 site_url: str = None, site_name: str = None,
                 **kwargs):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API ключ OpenRouter не найден. "
                "Установите переменную окружения OPENROUTER_API_KEY в .env файле."
            )

        # ВАЖНО: отключаем встроенный retry OpenAI, чтобы управлять им самим
        client_args = {
            "api_key": self.api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "max_retries": 0,
            "timeout": 120.0,
        }

        extra_headers = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-OpenRouter-Title"] = site_name
        if extra_headers:
            client_args["default_headers"] = extra_headers

        if proxy:
            logger.info(f"OpenRouter: Использование прокси: {proxy}")
            client_args["http_client"] = httpx.Client(proxy=proxy, timeout=120.0)

        self.client = OpenAI(**client_args)
        self.last_response_info = {}
        logger.info(f"OpenRouter: Инициализирован с моделью {model}")

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.01),
            "max_tokens": kwargs.get("max_tokens", 1024),
        }
        if "seed" in kwargs and kwargs["seed"] is not None:
            create_kwargs["seed"] = kwargs["seed"]

        last_exception = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            start_time = time.time()
            try:
                response = self.client.chat.completions.create(**create_kwargs)

                elapsed = time.time() - start_time
                content = response.choices[0].message.content
                usage = response.usage

                cost = getattr(usage, 'cost', None) if usage else None
                tokens = usage.completion_tokens if usage else None
                tokens_per_sec = tokens / elapsed if tokens else None

                self.last_response_info = {
                    "tokens": tokens,
                    "time_seconds": elapsed,
                    "tokens_per_sec": tokens_per_sec,
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                    "cost": cost,
                    "model": self.model,
                    "attempts": attempt
                }

                if attempt > 1:
                    logger.info(f"OpenRouter: Успех с {attempt}-й попытки")

                logger.debug(f"OpenRouter: {tokens} tokens in {elapsed:.2f}s "
                             f"({tokens_per_sec:.1f} tok/s)" if tokens_per_sec else "")
                return content

            except RateLimitError as e:
                last_exception = e
                # Пытаемся достать Retry-After из заголовков
                retry_after = self._extract_retry_after(e)
                delay = min(retry_after, self.MAX_DELAY)
                logger.warning(
                    f"OpenRouter: Rate limit (429). "
                    f"Попытка {attempt}/{self.MAX_RETRIES}. "
                    f"Ожидание {delay}с..."
                )
                time.sleep(delay)

            except (APITimeoutError, APIError) as e:
                last_exception = e
                # Экспоненциальный backoff для других API-ошибок
                delay = min(self.BASE_DELAY * (2 ** (attempt - 1)), self.MAX_DELAY)
                logger.warning(
                    f"OpenRouter: API ошибка ({type(e).__name__}). "
                    f"Попытка {attempt}/{self.MAX_RETRIES}. "
                    f"Ожидание {delay}с..."
                )
                time.sleep(delay)

            except Exception as e:
                # Не-API ошибки (например, сетевые) — не ретраим
                logger.error(f"OpenRouter: Неожиданная ошибка: {e}")
                raise

        logger.error(f"OpenRouter: Все {self.MAX_RETRIES} попыток исчерпаны.")
        raise last_exception

    @staticmethod
    def _extract_retry_after(exc: RateLimitError) -> int:
        """Извлекает Retry-After из заголовков или metadata, по умолчанию 30с."""
        # 1. Пробуем заголовки
        try:
            headers = getattr(exc, 'headers', {}) or {}
            ra = headers.get('Retry-After') or headers.get('retry-after')
            if ra:
                return int(ra)
        except Exception:
            pass

        # 2. Пробуем body.metadata
        try:
            body = getattr(exc, 'body', {}) or {}
            metadata = body.get('metadata', {}) or {}
            ra = metadata.get('retry_after_seconds')
            if ra:
                return int(ra)
        except Exception:
            pass

        # 3. Дефолт
        return 30