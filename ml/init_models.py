from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache

YOLO_MODEL_PATH = "yolo11s_best.pt"

@lru_cache(maxsize=1)
def get_text_box_detector():
    return YOLO(YOLO_MODEL_PATH, verbose=False)

@lru_cache(maxsize=1)
def get_text_recognizer():
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