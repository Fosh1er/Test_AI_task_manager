import torch
import gigaam
from pyannote.core import Segment
import whisperx
import numpy as np
from pyannote.audio import Pipeline

class Transcribator():
    def __init__(self):
        self.model = gigaam.load_model('v3_e2e_rnnt', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = Pipeline.from_pretrained("E:\Проекты\MVP_ASR_AI\hf_dependencies\config.yaml")
    def transcribe_audio(audio_path, self):
        # Получаем транскрипцию (список "отрезок речи" -> "текст")
        transcription = self.model.transcribe_longform(audio_path)
        return transcription

    def diarize_audio(audio_path,self):
        # Получаем разметку по спикерам
        diarization = self.pipeline(audio_path)
        # Возвращает список (спикер, время_начала, время_конца)
        return diarization

    def full_pipeline(self, audio_path):
        # 1. Получаем транскрипцию со словами и таймкодами
        asr_result = self.model.transcribe_longform(audio_path, return_timestamps='word')

        # 2. Получаем диаризацию (предполагаем, что self.diarization_pipeline уже инициализирован)
        diarization = self.diarization_pipeline(audio_path)  # возвращает pyannote.core.Annotation

        # 3. Преобразуем диаризацию в удобный список интервалов
        speaker_segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start': segment.start,
                'end': segment.end,
                'speaker': speaker
            })

        # 4. Сортируем сегменты по start (обычно уже отсортированы, но на всякий случай)
        speaker_segments.sort(key=lambda x: x['start'])

        # 5. Сопоставляем слова со спикерами
        segments_with_speakers = []

        for word_info in asr_result['words']:
            word_start = word_info['start_time']
            word_end = word_info.get('end_time', word_start + 0.1)  # если end_time нет, берём start+0.1с
            word_text = word_info['text']

            # Ищем сегмент, в который попадает слово
            speaker = None
            for seg in speaker_segments:
                if seg['start'] <= word_start < seg['end']:
                    speaker = seg['speaker']
                    break

            if speaker is None:
                speaker = "UNKNOWN"  # если слово вне сегментов (например, пауза между репликами)

            segments_with_speakers.append({
                'word': word_text,
                'start': word_start,
                'end': word_end,
                'speaker': speaker
            })

        # 6. (Опционально) Склеиваем слова одного спикера в предложения
        return segments_with_speakers