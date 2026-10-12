"""History-length buckets (PLAN.md §3b P1-1, P1-2).

Bucket length = the number of history events actually fed to the model at t_pred, i.e. the
capped length `EvalFrame.hist_len_capped` (min(full, 100)) that M3 stores in every frame and that
`eval.frame.history_bucket` maps to "0" / "1-5" / "6-20" / ">20". The same frame users (hence the
same buckets) are used for Pop, ALS and the three-tower model. The main (masked) population has
>= 1 history event by construction, so it is exactly covered by 1-5 / 6-20 / >20; the cold-user
frame is reported separately and is not bucketed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kuairand_rec.eval.frame import EvalFrame, history_bucket

BUCKETS = ("1-5", "6-20", ">20")


def frame_buckets(frame: EvalFrame) -> np.ndarray:
    """Per frame user bucket label (frame user order). Main frames must not contain bucket 0."""
    b = history_bucket(frame.hist_len_capped)
    if (b == "0").any():
        raise ValueError(f"{frame.split}/{frame.task}/{frame.kind}: users without history in a bucketed frame")
    if not np.array_equal(frame.hist_len_capped, np.minimum(frame.hist_len_full, 100)):
        raise ValueError("hist_len_capped is not min(hist_len_full, 100)")
    return b


def bucket_counts(frame: EvalFrame) -> pd.DataFrame:
    b = frame_buckets(frame)
    n = len(b)
    rows = [{"bucket": k, "n_users": int((b == k).sum()), "share": float((b == k).mean()) if n else 0.0,
             "gt_pairs": int(np.diff(frame.gt_indptr)[b == k].sum())} for k in BUCKETS]
    df = pd.DataFrame(rows)
    if int(df.n_users.sum()) != n:
        raise AssertionError("buckets do not partition the frame users")
    return df


def check_partition(frame: EvalFrame, per_user: pd.DataFrame, metrics) -> float:
    """Count-weighted bucket means must reproduce the overall mean (same users, same weights).
    Returns the max absolute deviation (asserted <= 1e-12)."""
    b = frame_buckets(frame)
    if not np.array_equal(per_user.user_id.to_numpy(), frame.users):
        raise ValueError("per-user table is not aligned with the frame")
    dev = 0.0
    for m in metrics:
        v = per_user[m].to_numpy(np.float64)
        w = sum(v[b == k].sum() for k in BUCKETS) / len(v)
        dev = max(dev, abs(w - v.mean()))
    if dev > 1e-12:
        raise AssertionError(f"bucket means do not reproduce the overall mean (max dev {dev})")
    return dev
