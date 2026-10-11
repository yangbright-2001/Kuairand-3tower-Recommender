"""Loader guards: forbidden/out-of-scope files are refused; metadata keys must be unique."""

import pandas as pd
import pytest

from kuairand_rec.data import io


def test_forbidden_and_excluded_files_refused(tmp_path):
    for name in ("video_features_statistic_pure.csv", "log_random_4_22_to_5_08_pure.csv"):
        with pytest.raises(ValueError):
            io._read(name, tmp_path)


def test_duplicate_metadata_key_raises(tmp_path):
    pd.DataFrame({"video_id": [0, 1, 1], "music_id": [5, 6, 7]}).to_csv(
        tmp_path / "video_features_basic_pure.csv", index=False
    )
    with pytest.raises(ValueError, match="not unique"):
        io.load_video_features(tmp_path)
