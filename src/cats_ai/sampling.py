import glob
from pathlib import Path

from cats_ai.config import N_PER_CLASS_CRASH, N_PER_CLASS_NORMAL, get_rng


def sample_generator(root, only_crash=False):
    """
    Generator for video files
    """
    for path_str in sorted(glob.iglob("**/*.mp4", root_dir=root, recursive=True)):
        rel_path = Path(path_str)
        full_path = root / rel_path

        if "Normal" in rel_path.parts:
            label = "normal"
        elif "Crash-1500" in rel_path.parts:
            label = "crash"

        # skips non crash videos
        if only_crash and label == "normal":
            continue

        yield full_path, label


def sample_generator_limited_percentage(root):
    """
    Yield a random balanced subset
    """

    _rng = get_rng()

    normal_paths = list((root / "Normal").glob("*.mp4"))
    crash_paths = list((root / "Crash-1500").glob("*.mp4"))

    normal_sample = _rng.sample(normal_paths, N_PER_CLASS_NORMAL)
    crash_sample = _rng.sample(crash_paths, N_PER_CLASS_CRASH)

    samples = [(path, "normal") for path in normal_sample] + [
        (path, "crash") for path in crash_sample
    ]

    _rng.shuffle(samples)

    yield from samples
