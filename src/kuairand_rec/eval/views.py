"""The only two ways a model may see events (PLAN.md §3c, information boundary).

* `train_view`   -> input to every fit (Pop counts, ALS factors). split == 0 only.
* `history_view` -> input to scoring / fold-in at a frozen prediction time:
                    val  -> t < T_val  (train events only)
                    test -> t < T_test (train + val events)
GT (events at t >= t_pred) is never reachable through either view.
"""

from __future__ import annotations

import pandas as pd

from kuairand_rec.data.catalog import assert_train_only

SPLIT_ID = {"val": 1, "test": 2}


def prediction_time(meta: dict, split_name: str) -> int:
    return {"val": int(meta["t_val_ms"]), "test": int(meta["t_test_ms"])}[split_name]


def train_view(events: pd.DataFrame) -> pd.DataFrame:
    tr = events[events.split == 0]
    assert_train_only(tr)
    return tr


def history_view(events: pd.DataFrame, split_name: str, meta: dict) -> pd.DataFrame:
    """Events strictly before the frozen prediction time of `split_name`."""
    t = prediction_time(meta, split_name)
    h = events[events.time_ms < t]
    # Split labels and time must agree: anything else means the cut or the split column is wrong.
    max_split = SPLIT_ID[split_name] - 1
    if len(h) and int(h.split.max()) > max_split:
        raise ValueError(f"history_view({split_name}) contains split > {max_split}")
    later = events[events.time_ms >= t]
    if len(later) and int(later.split.min()) <= max_split:
        raise ValueError(f"events of split <= {max_split} at or after t_pred for {split_name}")
    return h
