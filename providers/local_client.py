import ollama
import time
from logger import logger

class LocalProvider:
    def __init__(self, model: str = "qwen3.5:9b", host: str = None,
                 num_ctx: int = None, num_thread: int = None, num_gpu: int = None):
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.options = {}
        if num_ctx is not None:
            self.options["num_ctx"] = num_ctx
        if num_thread is not None:
            self.options["num_thread"] = num_thread
        if num_gpu is not None:
            self.options["num_gpu"] = num_gpu
        self.last_response_info = {}

    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        options = self.options.copy()
        options.update(kwargs.pop("options", {}))

        # Принудительно ставим temperature=0 для воспроизводимости
        kwargs.setdefault("temperature", 0.0)
        if "num_predict" not in options and "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]

        start = time.time()
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options=options,
            stream=False
        )
        elapsed = time.time() - start
        content = response['message']['content']

        eval_count = response.get("eval_count")
        eval_duration = response.get("eval_duration")
        tokens_per_sec = eval_count / (eval_duration / 1e9) if eval_count and eval_duration else None

        self.last_response_info = {
            "tokens": eval_count,
            "time_seconds": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }
        if tokens_per_sec:
            logger.debug(f"Ollama: {eval_count} tokens, {elapsed:.2f}s, {tokens_per_sec:.1f} tok/s")
        return content