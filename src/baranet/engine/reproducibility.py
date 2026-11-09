"""Seeding and provenance helpers for reproducible runs."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment_report(root: str | Path | None = None) -> str:
    lines = [
        f"python={os.sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_version={torch.version.cuda}",
        f"git_commit={git_commit()}",
    ]
    return "\n".join(lines) + "\n"
