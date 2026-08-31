import glob
import os
from pathlib import Path

import cv2
import pandas as pd
from agh_vqis import VQIs, process_folder_w_mm_files

from cats_ai.config import get_rng


def set_frames(video_path: str, n_frames: int, tmp_dir: Path):
    cap = cv2.VideoCapture(video_path)
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

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

    out = cv2.VideoCapture(out_path)
    print(int(out.get(cv2.CAP_PROP_FRAME_COUNT)))

    print(str(out_path))
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


def video_metrics(path="vids/"):
    # Desired final columns (optional, only used if we need an empty frame)
    columns = [
        "Frame",
        "Blockiness",
        "SA",
        "Letterbox",
        "Pillarbox",
        "Blockloss",
        "Blur",
        "TA",
        "Blackout",
        "Freezing",
        "Exposure(bri)",
        "Contrast",
        "Interlace",
        "Noise",
        "Slice",
        "Flickering",
        "UGC",
    ]

    # 1) Run your metric extraction
    process_folder_w_mm_files(
        Path(path),
        options={
            VQIs.colourfulness: False,
            VQIs.letterbox: False,
            VQIs.pillarbox: False,
            VQIs.blockloss: False,
            VQIs.blackout: False,
            VQIs.freezing: False,
            VQIs.slice: False,
            VQIs.flickering: False,
        },
    )

    # 2) Read all CSVs and compute mean per file
    dfs = []

    for path_str in sorted(glob.iglob("*.csv")):
        # Read CSV
        df = pd.read_csv(path_str)

        # Drop any index columns like Unnamed: 0, Unnamed: 0.1, ...
        df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

        # Keep Frame if you want an identifier, otherwise it will be averaged away
        # Compute mean of numeric columns only
        metrics = df.mean(numeric_only=True).to_frame().T

        # Optional: attach a video ID (e.g. filename) to know which row is which
        # metrics["Video"] = os.path.basename(path_str)

        dfs.append(metrics)

    # 3) Concatenate all per-file means
    if dfs:
        res_df = pd.concat(dfs, ignore_index=True)
    else:
        # No CSVs produced -> return empty frame with expected columns
        res_df = pd.DataFrame(columns=columns)

    # 4) Clean up intermediate CSVs
    for f in glob.glob("*.csv"):
        os.remove(f)

    # 5) Save final result without index, so no Unnamed: 0 next time
    res_df.to_csv("normal.csv", index=False)

    return res_df
