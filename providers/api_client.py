from openai import OpenAI
import time
import os
from logger import logger

class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.last_response_info = {}

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 1024),
            seed=kwargs.get("seed")  # для воспроизводимости
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