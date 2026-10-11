"""Small shared utilities: seeding and run metadata."""

from __future__ import annotations

import os
import random
import subprocess

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def git_sha() -> str:
    """Current git commit SHA (with '-dirty' suffix if the tree has changes)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
