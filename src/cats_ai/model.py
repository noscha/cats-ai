import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from cats_ai.config import HF_TOKEN, MODEL_ID_2B, MODEL_ID_4B, MODEL_ID_8B


def load_model_and_processor(model_id: str):

    match model_id:
        case "2B":
            model_id = MODEL_ID_2B
        case "4B":
            model_id = MODEL_ID_4B
        case "8B":
            model_id = MODEL_ID_8B

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        token=HF_TOKEN,
    )

    # Deactivate Learning
    model.eval()  # train()
    model.config.use_cache = True  # False
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()  # _enable()

    processor = AutoProcessor.from_pretrained(
        model_id,
        token=HF_TOKEN,
    )

    print(model.hf_device_map, flush=True)
    print(next(model.parameters()).device, next(model.parameters()).dtype, flush=True)

    return model, processor
