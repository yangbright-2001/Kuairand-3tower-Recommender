"""S9 explore configs: every config is registered with a kind, and its diff against its reference
is exactly what that kind allows (PLAN.md D1-10: one change at a time, <= 7 extra configs).

Kinds and their reference config:
  seed_calibration      vs threetower_u0.yaml: exactly {train.seed}
  d1_10_*               vs threetower_u0.yaml: exactly one registered knob (a hyperparameter variant)
  seed_replicate        vs configs/explore/<base>.yaml (an approved d1_10_* variant): exactly {train.seed};
                        hence vs threetower_u0.yaml: the base's knob + train.seed
  reproduction_control  vs threetower_u0.yaml (or `reference: <formal config stem>`): nothing (only exp / run_root)
A seed_replicate may instead have `base: formal:<config stem>` (a formal ablation config such as
threetower_abl_nomusic.yaml): then exactly {train.seed} vs that formal config. Formal ablations are
not D1-10 variants, so this neither consumes nor dodges the budget.
Every config, whatever its kind, differs from its reference in exp and run_root."""

from __future__ import annotations

import yaml

from kuairand_rec import paths
from kuairand_rec.models import features as F
from kuairand_rec.models import train as TR
from kuairand_rec.models.towers import TowerConfig

CFG = paths.PROJECT_ROOT / "configs"
EXPLORE = CFG / "explore"
KINDS = {"seed_calibration", "seed_replicate", "reproduction_control"}


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


def _vs_u0(name: str) -> dict:
    """Training-relevant diff vs the formal U0 config ({key: new value}); exp / run_root checked here."""
    u0 = _load(CFG / "threetower_u0.yaml")
    d = _flat_diff(u0, _load(EXPLORE / f"{name}.yaml"))
    assert d.pop("exp") == ("u0", name)
    assert d.pop("run_root") == (u0["run_root"], "/local-scratch/liang/runs_explore")
    return {k: v[1] for k, v in d.items()}


FORMAL_ABLATIONS = {"threetower_abl_nomusic": {"model.use_music": False}}


def _vs_formal(name: str, stem: str) -> dict:
    """Diff vs a formal config ({key: new value}); exp / run_root checked here."""
    ref = _load(CFG / f"{stem}.yaml")
    d = _flat_diff(ref, _load(EXPLORE / f"{name}.yaml"))
    assert d.pop("exp") == (ref["exp"], name)
    assert d.pop("run_root") == (ref["run_root"], "/local-scratch/liang/runs_explore")
    return {k: v[1] for k, v in d.items()}


def test_formal_ablation_references_are_single_switch_vs_u0():
    for stem, switch in FORMAL_ABLATIONS.items():
        u0, ab = _load(CFG / "threetower_u0.yaml"), _load(CFG / f"{stem}.yaml")
        d = _flat_diff(u0, ab)
        d.pop("exp")
        assert {k: v[1] for k, v in d.items()} == switch


def test_registry_matches_config_files_and_budget():
    reg = _load(EXPLORE / "registry.yaml")
    runs = reg["runs"]
    files = {p.stem for p in EXPLORE.glob("x_*.yaml")}
    assert files == set(runs), "every explore config must be registered (and vice versa)"
    for name, r in runs.items():
        assert r["kind"] in KINDS or r["kind"].startswith("d1_10_"), name
        # Only hyperparameter variants consume the D1-10 budget; nothing else may claim or dodge it.
        assert bool(r["d1_10"]) == r["kind"].startswith("d1_10_"), name
    assert sum(bool(r["d1_10"]) for r in runs.values()) <= reg["d1_10_budget"] == 7


def test_hyperparameter_variants_and_seed_calibrations_change_exactly_one_key_vs_u0():
    for name, r in _load(EXPLORE / "registry.yaml")["runs"].items():
        if r["kind"].startswith("d1_10_") or r["kind"] == "seed_calibration":
            assert len(r["change"]) == 1, "one change at a time (D1-10)"
            assert _vs_u0(name) == r["change"], name
            if r["kind"] == "seed_calibration":
                assert set(r["change"]) == {"train.seed"}
            else:
                assert "train.seed" not in r["change"], "a variant must keep U0's seed"


def test_seed_replicates_change_only_the_seed_vs_their_approved_variant():
    runs = _load(EXPLORE / "registry.yaml")["runs"]
    for name, r in runs.items():
        if r["kind"] != "seed_replicate" or r["base"].startswith("formal:"):
            continue
        base = runs[r["base"]]
        assert base["kind"].startswith("d1_10_"), "a replicate must replicate a hyperparameter variant"
        assert set(r["change"]) == {"train.seed"} and len(r["change"]) == 1
        d = _flat_diff(_load(EXPLORE / f"{r['base']}.yaml"), _load(EXPLORE / f"{name}.yaml"))
        assert d.pop("exp") == (r["base"], name)
        assert {k: v[1] for k, v in d.items()} == r["change"], name
        assert r["change"]["train.seed"] != _load(EXPLORE / f"{r['base']}.yaml")["train"]["seed"]
        # Consequently, vs U0: exactly the base's single knob + the seed.
        assert _vs_u0(name) == {**base["change"], **r["change"]}


def test_seed_replicates_of_formal_ablations_change_only_the_seed():
    for name, r in _load(EXPLORE / "registry.yaml")["runs"].items():
        if r["kind"] == "seed_replicate" and r["base"].startswith("formal:"):
            stem = r["base"].split(":", 1)[1]
            assert stem in FORMAL_ABLATIONS, f"{name}: unknown formal base {stem}"
            assert set(r["change"]) == {"train.seed"} and len(r["change"]) == 1
            assert _vs_formal(name, stem) == r["change"], name
            assert r["change"]["train.seed"] != _load(CFG / f"{stem}.yaml")["train"]["seed"]
            assert _vs_u0(name) == {**FORMAL_ABLATIONS[stem], **r["change"]}


def test_reproduction_control_equals_its_reference_training_config():
    for name, r in _load(EXPLORE / "registry.yaml")["runs"].items():
        if r["kind"] == "reproduction_control":
            assert r["change"] == {}, name
            ref = r.get("reference")
            if ref is None:
                assert _vs_u0(name) == {}, name
            else:
                assert ref in FORMAL_ABLATIONS and _vs_formal(name, ref) == {}, name


def _allowed_model(name: str) -> TowerConfig:
    """Default D1-1 tower, except runs whose registered reference is a formal ablation."""
    r = _load(EXPLORE / "registry.yaml")["runs"][name]
    stem = r.get("reference") or (r["base"].split(":", 1)[1] if str(r.get("base", "")).startswith("formal:") else None)
    if stem is None:
        return TowerConfig()
    return TowerConfig.from_dict(_load(CFG / f"{stem}.yaml")["model"])


def test_every_explore_config_keeps_the_d1_protocol():
    for p in EXPLORE.glob("x_*.yaml"):
        c = _load(p)
        tc = TR.TrainConfig.from_dict(c["train"])
        assert TowerConfig.from_dict(c["model"]) == _allowed_model(p.stem)
        assert (tc.lr, tc.tau, tc.warmup_steps, tc.weight_decay, tc.clip_grad_norm) == \
            (_load(EXPLORE / f"{p.stem}.yaml")["train"]["lr"], _load(EXPLORE / f"{p.stem}.yaml")["train"]["tau"], 300, 0.01, 1.0)
        assert F.HistoryConfig.from_dict(c["history"]) == F.HistoryConfig()
        assert c["require_clean_git"] is True and c["condition"] == "normal" and c["user_features"] == "u0"
        assert (tc.batch_size, tc.max_epochs, tc.patience, tc.like_oversample) == (1024, 30, 3, 1)
        assert c["eval"] == _load(CFG / "threetower_u0.yaml")["eval"]


def test_explore_run_root_is_not_the_formal_root():
    for p in EXPLORE.glob("x_*.yaml"):
        rr = _load(p)["run_root"].rstrip("/")
        assert rr != "/local-scratch/liang/runs" and rr.endswith("runs_explore")
