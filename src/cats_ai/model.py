import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cats_ai.config import MODEL_ID


def load_model_and_processor(model_id: str = MODEL_ID):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        token=os.environ["HF_API_KEY"],
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=os.environ["HF_API_KEY"],
    )

    return model, processor