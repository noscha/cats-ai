import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cats_ai.config import MODEL_ID, HF_TOKEN


def load_model_and_processor(model_id: str = MODEL_ID):

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        token=HF_TOKEN,
    )

    # Deactivate Learning
    model.eval()  # train()
    model.config.use_cache = True # False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable() # _enable()

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=HF_TOKEN,
    )

    return model, processor