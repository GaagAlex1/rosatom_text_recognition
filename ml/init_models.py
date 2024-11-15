import os
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolo11s_best.pt')

@lru_cache(maxsize=1)
def get_text_box_detector():
    """
    Возвращает объект класса YOLO - модель для детекции текстовых блоков

    Returns:
    YOLO: объект класса YOLO для детекции текстовых блоков
    """
    return YOLO(YOLO_MODEL_PATH, verbose=False)


@lru_cache(maxsize=1)
def get_text_recognizer():
    """
    Возвращает GOT-OCR2_0 - трансформер для распознавания текста на изображении

    Returns:
    AutoModel: GOT-OCR2_0
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "ucaslcl/GOT-OCR2_0", trust_remote_code=True
    )
    text_recognizer = AutoModel.from_pretrained(
        "ucaslcl/GOT-OCR2_0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    text_recognizer = text_recognizer.eval().cuda()
    text_recognizer.generation_config.pad_token_id = tokenizer.pad_token_id
    return text_recognizer, tokenizer