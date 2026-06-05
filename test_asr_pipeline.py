import os
# Жестко фиксируем переменные окружения ДО импорта любых зависимостей
os.environ["PYCTCDECODE_CACHE"] = "C:/hf_cache"
os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

from mini_diarizator import Diarizator
from transcribator import Transcribator

AUDIO_PATH = "audio/discord.wav"
OUTPUT_TXT = "transcripts/transcript_discord.txt"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = "cpu"
SPEAKER_MAP = {}

def main():
    print("Инициализация диаризатора...")
    diar = Diarizator(hf_token=HF_TOKEN, device=DEVICE)

    print("Инициализация транскрибатора...")
    trans = Transcribator(device=DEVICE, use_lm=True, cache_dir="C:/hf_cache")

    print(f"Диаризация файла {AUDIO_PATH}...")
    segments = diar.diarize(AUDIO_PATH)

    print("Транскрипция и форматирование...")
    transcript = trans.transcribe_segments(
        audio_path=AUDIO_PATH,
        segments=segments,
        speaker_map=SPEAKER_MAP,
        output_txt=OUTPUT_TXT
    )

    print("\n" + "="*50)
    print("ИТОГОВАЯ ТРАНСКРИПЦИЯ:")
    print("="*50)
    print(transcript)

if __name__ == "__main__":
    main()