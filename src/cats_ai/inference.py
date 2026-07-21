from sympy.physics.quantum.trace import Tr
import torch
from cats_ai.config import MAX_NEW_TOKENS
from qwen_vl_utils import process_vision_info
from cats_ai.messages import build_messages
from cats_ai.video import get_last_non_accident_frame, set_frames

from datetime import datetime


def query(video_path, prompt, model, processor, crash_masking=False, tmp_dir=None):
    """
    Query a video
    """
    last_non_accident_frame = float("inf")

    if crash_masking:
        last_non_accident_frame = get_last_non_accident_frame(video_path)
        video_path = set_frames(video_path, last_non_accident_frame, tmp_dir)

    messages = build_messages(video_path, prompt, last_non_accident_frame)
    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Build message done", flush=True)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} apply_chat_template done", flush=True)

    images, videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} process_vision_info done", flush=True)

    videos, video_metadata = zip(*videos)
    videos = list(videos)
    video_metadata = list(video_metadata)

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        return_tensors="pt",
        **video_kwargs,
    ).to(model.device)

    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} processor done", flush=True)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Inference done", flush=True)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} response done", flush=True)

    return response
