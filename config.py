import os

# ---- Провайдер и модель ----
PROVIDER_MODE = os.getenv("LLM_PROVIDER", "local")   # "local" или "api"
MODEL_NAME = os.getenv("LLM_MODEL", "qwen3.5:9b")   # имя модели в Ollama или OpenAI

# ---- Параметры для локальной модели (Ollama) ----
LLM_KWARGS = {
    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
    "num_thread": int(os.getenv("OLLAMA_NUM_THREAD", "4")),
    "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "-1")),   # -1 = все слои на GPU
}

# Если используется OpenAI API, можно задать модель и ключ здесь
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---- Параметры чанкования ----
CHUNK_SIZE = 12000
CHUNK_OVERLAP = 1000

# ---- Пути ----
TRANSCRIPTS_DIR = "transcripts"