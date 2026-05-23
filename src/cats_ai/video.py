import cv2
from pathlib import Path
from cats_ai.config import get_rng

def set_frames(video_path: str, n_frames: int, tmp_dir: Path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stem = Path(video_path).stem
    out_path = tmp_dir / f"{stem}_first_{n_frames}.mp4"

    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

    return str(out_path)


def get_last_non_accident_frame(video_path: str):

    if "Normal" in video_path:
        # Under 3 frames video pipe breaks
        _rng = get_rng()
        return _rng.randint(3, 50)

    with open("CarCrash/videos/Crash-1500.txt", "r", encoding="utf-8") as f:
        content = f.readlines()

    index = int(Path(video_path).stem) - 1
    start = content[index].index("[")
    end = content[index].index("]", start)
    lst = [int(x.strip()) for x in content[index][start + 1 : end].split(",")]

    return lst.index(1)