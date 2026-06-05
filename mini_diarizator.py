import os
import torch
from typing import Optional
from pyannote.audio import Pipeline


class Diarizator:
    def __init__(self, hf_token: Optional[str] = None, device: str = "cuda"):
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        print("Загрузка pyannote/speaker-diarization-3.1...")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        if device == "cuda" and torch.cuda.is_available():
            self.pipeline = self.pipeline.to(torch.device("cuda"))
        print("Диаризатор готов.")

    def diarize(self, audio_path: str) -> list:
        import soundfile as sf

        print(f"Загрузка аудио {audio_path}...")
        data, sample_rate = sf.read(audio_path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        waveform = torch.from_numpy(data.T).float()

        print("Запуск диаризации...")
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # В версии 3.1+ результат оборачивается в DiarizeOutput,
        # а нужная аннотация находится в атрибуте speaker_diarization
        if hasattr(output, 'speaker_diarization'):
            annotation = output.speaker_diarization
        elif hasattr(output, 'annotation'):
            annotation = output.annotation
        elif hasattr(output, 'diarization'):
            annotation = output.diarization
        else:
            annotation = output  # fallback

        segments = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": segment.start,
                "end": segment.end
            })

        print(f"Диаризация завершена. Найдено сегментов: {len(segments)}")
        return segments