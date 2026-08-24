import json


def build_messages(
    video_path,
    prompt,
    last_non_accident_frame,
    sft_target=None,
):

    NFRAMES = 50  # TODO fix
    if video_path in [
        "CarCrash/videos/Normal/000615.mp4",
        "CarCrash/videos/Normal/002058.mp4",
    ]:
        NFRAMES = 49

    nframes = min(NFRAMES, last_non_accident_frame)
    nframes -= nframes % 2  # qwen_vl_utils uses multiple of two for grouping

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": nframes,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    if sft_target is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(sft_target, ensure_ascii=False),
                    }
                ],
            }
        )

    return messages
