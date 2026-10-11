"""Read-only loaders for the four agreed KuaiRand-Pure raw files.

Only the two standard logs, user_features_pure.csv and video_features_basic_pure.csv
may be loaded. The statistic file (future leakage) and the random log (out of scope)
are refused explicitly so they cannot slip into the pipeline by accident.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kuairand_rec import paths

ALLOWED_FILES = (*paths.STANDARD_LOGS, paths.USER_FEATURES, paths.VIDEO_FEATURES_BASIC)


def _read(name: str, raw_dir: Path) -> pd.DataFrame:
    if name not in ALLOWED_FILES:
        raise ValueError(f"{name} is not an allowed input (forbidden or out of scope)")
    return pd.read_csv(Path(raw_dir) / name)


def load_standard_logs(raw_dir: Path = paths.RAW_DIR) -> pd.DataFrame:
    """Concatenate the two standard logs. Adds `src_file` (0 or 1); no rows are dropped."""
    frames = []
    for i, name in enumerate(paths.STANDARD_LOGS):
        d = _read(name, raw_dir)
        d["src_file"] = np.int8(i)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def _assert_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    # A non-unique metadata key would silently duplicate log rows on join.
    n_dup = int(df[key].duplicated().sum())
    if n_dup or df[key].isna().any():
        raise ValueError(f"{name}: primary key {key} not unique/non-null ({n_dup} duplicates)")


def load_user_features(raw_dir: Path = paths.RAW_DIR) -> pd.DataFrame:
    df = _read(paths.USER_FEATURES, raw_dir)
    _assert_unique_key(df, "user_id", paths.USER_FEATURES)
    return df


def load_video_features(raw_dir: Path = paths.RAW_DIR) -> pd.DataFrame:
    df = _read(paths.VIDEO_FEATURES_BASIC, raw_dir)
    _assert_unique_key(df, "video_id", paths.VIDEO_FEATURES_BASIC)
    return df
