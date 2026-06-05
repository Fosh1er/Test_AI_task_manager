import voicetag
import resemblyzer
from voicetag import VoiceTag
import os
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import torch

class Diarizator:
    """
    Класс для диаризации аудио.
    Умеет загружать модели pyannote.audio из кэша или скачивать их при необходимости.
    """
    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None):
        """
        Инициализация диаризатора.
        Args:
            device: Устройство для вычислений ("cuda" или "cpu")
            hf_token: Токен для доступа к gated-моделям pyannote на Hugging Face.
                       Необходим для первой загрузки моделей.
        """
        self.device = device
        self.hf_token = os.getenv("HF_TOKEN")

        # 1. Устанавливаем переменную окружения, если передан токен
        # Это поможет pyannote.audio и другим библиотекам найти его
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token

        # 2. Инициализируем модель диаризации
        # from_pretrained автоматически проверит кэш и скачает модель, если её там нет
        print("Загрузка модели диаризации pyannote/speaker-diarization-3.1...")
        try:
            from pyannote.audio import Pipeline
            # Кэширование встроено в from_pretrained.
            # Для полной гарантии оффлайн-работы после первой загрузки можно добавить параметр local_files_only=True
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )
            # Переносим пайплайн на нужное устройство
            if torch.cuda.is_available() and device == "cuda":
                self.pipeline = self.pipeline.to(torch.device("cuda"))
            print("Модель диаризации загружена.")
        except Exception as e:
            print(f"Ошибка при загрузке модели диаризации: {e}")
            raise

    def diarize(self, audio_path: str) -> list:
        import soundfile as sf
        import torch

        # Загрузка аудио
        data, sample_rate = sf.read(audio_path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        waveform = torch.from_numpy(data.T).float()

        # Диаризация
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # Извлекаем аннотацию (основную)
        annotation = output.speaker_diarization  # <-- вот здесь обращаемся к атрибуту

        segments = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end
            })
        return segments