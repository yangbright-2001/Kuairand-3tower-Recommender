"""S9 explore configs: each differs from the formal U0 config only in `exp`, `run_root` and its
single registered knob, and the registered D1-10 count stays within the budget (PLAN.md D1-10)."""

from __future__ import annotations

import yaml

from kuairand_rec import paths
from kuairand_rec.models import features as F
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TowerConfig

CFG = paths.PROJECT_ROOT / "configs"
EXPLORE = CFG / "explore"


def _load(p):
    return yaml.safe_load(p.read_text())


def _flat_diff(a: dict, b: dict, pre: str = "") -> dict:
    out = {}
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out.update(_flat_diff(x, y, f"{pre}{k}."))
        elif x != y:
            out[f"{pre}{k}"] = (x, y)
    return out


def test_registry_matches_config_files_and_budget():
    reg = _load(EXPLORE / "registry.yaml")
    files = {p.stem for p in EXPLORE.glob("x_*.yaml")}
    assert files == set(reg["runs"]), "every explore config must be registered (and vice versa)"
    assert sum(bool(r["d1_10"]) for r in reg["runs"].values()) <= reg["d1_10_budget"] == 7
    assert all(not r["d1_10"] for r in reg["runs"].values() if r["kind"] == "seed_calibration")


def test_explore_configs_differ_from_u0_only_in_registered_knob():
    u0 = _load(CFG / "threetower_u0.yaml")
    reg = _load(EXPLORE / "registry.yaml")["runs"]
    for name, r in reg.items():
        c = _load(EXPLORE / f"{name}.yaml")
        d = _flat_diff(u0, c)
        assert d.pop("exp") == ("u0", name)
        assert d.pop("run_root") == (u0["run_root"], "/local-scratch/liang/runs_explore")
        assert {k: v[1] for k, v in d.items()} == r["change"], name
        assert len(r["change"]) == 1, "one change at a time (D1-10)"
        # The changed config still parses and keeps every other D1 setting.
        tc = TR.TrainConfig.from_dict(c["train"])
        assert TowerConfig.from_dict(c["model"]) == TowerConfig()
        assert F.HistoryConfig.from_dict(c["history"]) == F.HistoryConfig()
        assert c["require_clean_git"] is True and c["condition"] == "normal" and c["user_features"] == "u0"
        assert (tc.batch_size, tc.max_epochs, tc.patience, tc.like_oversample) == (1024, 30, 3, 1)


def test_explore_run_root_is_not_the_formal_root():
    for p in EXPLORE.glob("x_*.yaml"):
        rr = _load(p)["run_root"].rstrip("/")
        assert rr != "/local-scratch/liang/runs" and rr.endswith("runs_explore")
