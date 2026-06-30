import random
from pathlib import Path
import os

import numpy as np
import torch

SEED = 42
_RNG = random.Random(SEED + 1)

ROOT = Path("CarCrash/videos")
NORMAL_DIR = ROOT / "Normal"
CRASH_DIR = ROOT / "Crash-1500"
NFRAMES = 50
N_PER_CLASS_CRASH = 150  # 10% of data
N_PER_CLASS_NORMAL = 300  # 10% of data

OUTPUT_ROOT = Path("outputs")
MODEL_OUTPUT_PATH = OUTPUT_ROOT / "trials"
SFT_PATH = OUTPUT_ROOT / "sft"

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
HF_TOKEN = open("token.txt", "r").readline()
MAX_NEW_TOKENS = 500  # 400 and above for analysis, 64 for accident detection


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed common randomness sources.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng():
    return _RNG
