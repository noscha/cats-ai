from cats_ai.config import ROOT, N_PER_CLASS_NORMAL, N_PER_CLASS_CRASH, get_rng
from pathlib import Path
import glob

def sample_generator():
    """
    Generator for video files
    """
    for path_str in glob.iglob("**/*.mp4", root_dir=ROOT, recursive=True):
        rel_path = Path(path_str)
        full_path = ROOT / rel_path

        if "Normal" in rel_path.parts:
            label = "normal"
        elif "Crash-1500" in rel_path.parts:
            label = "crash"

        yield full_path, label


def sample_generator_limited():
    """
    Yield a random balanced subset
    """

    _rng = get_rng()

    normal_paths = list((ROOT / "Normal").glob("*.mp4"))
    crash_paths = list((ROOT / "Crash-1500").glob("*.mp4"))

    normal_sample = _rng.sample(normal_paths, N_PER_CLASS_NORMAL)
    crash_sample = _rng.sample(crash_paths, N_PER_CLASS_CRASH)

    samples = [(path, "normal") for path in normal_sample] + [
        (path, "crash") for path in crash_sample
    ]

    _rng.shuffle(samples)

    yield from samples