import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cats_ai.config import MODEL_ID


def load_model_and_processor(model_id: str = MODEL_ID, token: str | None = None):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        token=token,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=token,
    )

    return model, processor