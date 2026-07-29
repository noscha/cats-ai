import json

from cats_ai.config import NFRAMES


def build_messages(
    video_path,
    prompt,
    last_non_accident_frame,
    sft_target=None,
):


    if video_path in ["CarCrash/videos/Normal/000615.mp4", "CarCrash/videos/Normal/002058.mp4"]:
        NFRAMES = 49

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "nframes": min(NFRAMES, last_non_accident_frame - 1),
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
