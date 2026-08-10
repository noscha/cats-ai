import os
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42
_RNG = random.Random(SEED + 1)

ROOT = Path("CarCrash/videos")
ROOT_CLUSTER = Path("/scratch/cats-ai/CarCrash/videos")
NFRAMES = 50
N_PER_CLASS_CRASH = 150  # 10% of data
N_PER_CLASS_NORMAL = 300  # 10% of data

OUTPUT_ROOT = Path("outputs")
MODEL_OUTPUT_PATH = OUTPUT_ROOT / "trials"
SFT_PATH = OUTPUT_ROOT / "sft"

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"  # 2B and 8B are the only feasible models for our hardware
HF_TOKEN = open("token.txt", "r").readline()
MAX_NEW_TOKENS = 500  # 500 and above for analysis, 100 for accident detection


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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_rng():
    return _RNG
