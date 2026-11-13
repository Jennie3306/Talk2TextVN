# transcribe.py
# Chức năng: chuyển audio thành text (PhoWhisper hoặc Whisper gốc), log latency

import json
import os
import time
from datetime import datetime

import numpy as np

from config import (
    ASR_BACKEND,
    LANGUAGE,
    PHOWHISPER_MODEL,
    RESULTS_DIR,
    SAMPLE_RATE,
    WHISPER_MODEL,
)

# Cache model theo (backend, tên model) để không load lại mỗi lần gọi
_models = {}

# Prompt giúp Whisper gốc hiểu context tiếng Việt (PhoWhisper không cần)
_WHISPER_PROMPT = (
    "Đây là đoạn hội thoại tiếng Việt thông thường. "
    "Người nói có thể đề cập đến công nghệ, lập trình, "
    "trí tuệ nhân tạo, học máy, và các chủ đề hàng ngày."
)


def _default_model(backend: str) -> str:
    return PHOWHISPER_MODEL if backend == "phowhisper" else WHISPER_MODEL


def get_model(backend: str = ASR_BACKEND, model_name: str | None = None):
    """Load model một lần rồi cache lại."""
    model_name = model_name or _default_model(backend)
    key = (backend, model_name)
    if key in _models:
        return _models[key]

    print(f"Đang load {backend} model '{model_name}' (lần đầu sẽ tự download)...")
    if backend == "phowhisper":
        import torch
        from transformers import pipeline

        _models[key] = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,  # audio dài hơn 30s được cắt thành nhiều đoạn
        )
    elif backend == "whisper":
        import whisper

        _models[key] = whisper.load_model(model_name)
    else:
        raise ValueError(f"ASR_BACKEND không hợp lệ: {backend}")
    print("Load model xong!")
    return _models[key]


def _load_audio(audio_input) -> np.ndarray:
    """Nhận đường dẫn file hoặc numpy float32 16 kHz, trả về numpy float32."""
    if isinstance(audio_input, np.ndarray):
        return audio_input.astype(np.float32)
    from preprocess import preprocess

    return preprocess(audio_input)


def transcribe(audio_input, model_name: str | None = None, backend: str = ASR_BACKEND) -> dict:
    """
    Chuyển audio thành text.
    - audio_input: đường dẫn file .wav (str) hoặc numpy array float32 16 kHz
    - return: dict chứa raw_text, latency_s, model, backend, timestamp
    """
    model_name = model_name or _default_model(backend)
    model = get_model(backend, model_name)

    print("Đang nhận dạng giọng nói...")
    start_time = time.time()

    if backend == "phowhisper":
        audio = _load_audio(audio_input)
        result = model({"raw": audio, "sampling_rate": SAMPLE_RATE})
        raw_text = result["text"].strip()
    else:
        result = model.transcribe(
            audio_input,
            language=LANGUAGE,
            verbose=False,
            initial_prompt=_WHISPER_PROMPT,
        )
        raw_text = result["text"].strip()

    latency = round(time.time() - start_time, 3)
    output = {
        "raw_text": raw_text,
        "latency_s": latency,
        "model": model_name,
        "backend": backend,
        "language": LANGUAGE,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    print(f"Nhận dạng xong! ({latency}s): {raw_text}")
    return output


def transcribe_and_save(audio_input, model_name: str | None = None, backend: str = ASR_BACKEND) -> dict:
    """Transcribe và lưu kết quả ra file JSON trong results/."""
    result = transcribe(audio_input, model_name, backend)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, f"transcription_{result['timestamp']}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu kết quả: {save_path}")
    return result
