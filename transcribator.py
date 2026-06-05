import os
from typing import List, Dict, Optional
import torch
import soundfile as sf


class Transcribator:
    def __init__(self, device: str = "cuda", use_lm: bool = True, cache_dir: str = None):
        self.device = device
        self.use_lm = use_lm

        if cache_dir is None:
            cache_dir = "C:/hf_cache"
        os.environ["PYCTCDECODE_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir

        print("Загрузка GigaAM v3 CTC...")
        from transformers import AutoProcessor, AutoModelForCTC

        model_id = "waveletdeboshir/gigaam-v3-ctc-with-lm" if use_lm else "waveletdeboshir/gigaam-v3-ctc"

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model = AutoModelForCTC.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir
        )
        self.model.eval()
        self.model.to(self.device)
        print("Транскрибатор готов.")

    def transcribe_segments(self, audio_path: str, segments: List[Dict],
                            speaker_map: Optional[Dict[str, str]] = None,
                            output_txt: Optional[str] = None) -> str:
        if not segments:
            return ""

        # ОПТИМИЗАЦИЯ: Читаем файл целиком один раз, чтобы не долбить диск в цикле
        print("Чтение аудиофайла в память...")
        data, sr = sf.read(audio_path)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        full_waveform = torch.from_numpy(data.T).float()

        transcribed_segments = []
        total = len(segments)
        for i, seg in enumerate(segments):
            print(f"  Транскрибация сегмента {i + 1}/{total}...")
            text = self._transcribe_segment(full_waveform, sr, seg["start"], seg["end"])

            speaker = seg["speaker"]
            if speaker_map and speaker in speaker_map:
                speaker = speaker_map[speaker]

            transcribed_segments.append({
                "speaker": speaker,
                "text": text,
                "start": seg["start"],
                "end": seg["end"]
            })

        # Группировка последовательных реплик одного спикера
        merged = []
        current_speaker = None
        current_text_parts = []
        for ts in transcribed_segments:
            if ts["speaker"] != current_speaker:
                if current_speaker is not None:
                    merged.append(f"{current_speaker}: {' '.join(current_text_parts)}")
                current_speaker = ts["speaker"]
                current_text_parts = [ts["text"]]
            else:
                current_text_parts.append(ts["text"])

        if current_speaker is not None:
            merged.append(f"{current_speaker}: {' '.join(current_text_parts)}")

        result = "\n".join(merged)
        if output_txt:
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Результат сохранён в {output_txt}")
        return result

    def _transcribe_segment(self, waveform: torch.Tensor, sr: int, start_sec: float, end_sec: float) -> str:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr)
        segment = waveform[:, start_sample:end_sample]
        return self._transcribe_tensor(segment, sr)

    def _transcribe_tensor(self, waveform: torch.Tensor, sample_rate: int) -> str:
        if sample_rate != 16000:
            import torchaudio
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        if self.use_lm:
            text = self.processor.batch_decode(
                logits=logits.cpu().numpy(),
                beam_width=64,
                alpha=0.5,
                beta=0.5
            ).text[0]
        else:
            ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(ids)[0]
        return text.strip()