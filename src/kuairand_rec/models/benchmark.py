"""Training-only throughput benchmark (PLAN.md §3d D1-11 "single-GPU smoke and benchmark").

    uv run python -m kuairand_rec.models.benchmark --config configs/threetower_u0.yaml --steps 50
    uv run python -m kuairand_rec.models.benchmark --config configs/threetower_u0.yaml --steps 300 --val_once

Reuses the real `Trainer` (same samples, batches, loss, optimizer, fail-fast checks); it only
stops after `--steps` optimizer steps and skips validation / selection / export / test / parity.
Outputs `benchmark.json` in its own run directory (outside the worktree). The loss values it
records come from a few hundred warmup steps of an unselected model: they are a stability
check, NOT a model metric, and must never be reported as results.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from kuairand_rec import paths
from kuairand_rec.data.pipeline import load_processed
from kuairand_rec.models import data as D
from kuairand_rec.models import features as F
from kuairand_rec.models.towers import TowerConfig
from kuairand_rec.models.train import (
    TrainConfig, Trainer, _resolve, assert_clean_worktree, assert_outside_worktree, train_step, user_features_for,
)
from kuairand_rec.utils import git_sha

NOTE = "benchmark only: warmup-phase loss of an unselected model, not a model metric"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_steps(tr: Trainer, steps: int, skip: int = 5) -> dict:
    """Run `steps` real training steps with the Trainer's own batches; time data vs compute.

    `skip` first steps (CUDA context / allocator / kernel warm-up) are excluded from the rates.
    """
    dev = tr.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    recs, epoch = [], 0
    batches = iter(D.epoch_batches(len(tr.samples), tr.tc.batch_size, tr.tc.seed, epoch))
    for s in range(steps):
        idx = next(batches, None)
        if idx is None:                       # benchmark longer than an epoch: next permutation
            epoch += 1
            batches = iter(D.epoch_batches(len(tr.samples), tr.tc.batch_size, tr.tc.seed, epoch))
            idx = next(batches)
        _sync(dev)
        t0 = time.perf_counter()
        batch = tr.builder(idx, dev)
        _sync(dev)
        t1 = time.perf_counter()
        rec = train_step(tr.model, tr.opt, tr.sched, tr.items, batch, tr.tc, tr.step, tr._debug_dir())
        _sync(dev)
        t2 = time.perf_counter()
        tr.step += 1
        rec.update(data_s=t1 - t0, compute_s=t2 - t1, step_s=t2 - t0, batch_rows=len(idx))
        recs.append(rec)
    timed = recs[skip:] if len(recs) > skip else recs
    step_s = np.array([r["step_s"] for r in timed])
    data_s = np.array([r["data_s"] for r in timed])
    comp_s = np.array([r["compute_s"] for r in timed])
    steps_per_epoch = math.ceil(len(tr.samples) / tr.tc.batch_size)
    losses = np.array([r["loss"] for r in recs])
    gns = np.array([r["grad_norm"] for r in recs])
    out = {
        "steps": steps, "timed_steps": len(timed), "skipped_warmup_steps": len(recs) - len(timed),
        "batch_size": tr.tc.batch_size, "samples_per_epoch": len(tr.samples), "steps_per_epoch": steps_per_epoch,
        "steps_per_s": float(1.0 / step_s.mean()),
        "step_s": {"mean": float(step_s.mean()), "p50": float(np.median(step_s)), "p90": float(np.quantile(step_s, 0.9))},
        "data_s_mean": float(data_s.mean()), "compute_s_mean": float(comp_s.mean()),
        "data_share": float(data_s.sum() / step_s.sum()),
        "epoch_s_estimate": float(step_s.mean() * steps_per_epoch),
        "loss": {"first": float(losses[0]), "last": float(losses[-1]), "min": float(losses.min()),
                 "max": float(losses.max()), "all_finite": bool(np.isfinite(losses).all())},
        "grad_norm_pre_clip": {"first": float(gns[0]), "last": float(gns[-1]), "mean": float(gns.mean()),
                               "max": float(gns.max()), "all_finite": bool(np.isfinite(gns).all()),
                               "share_clipped": float((gns > tr.tc.clip_grad_norm).mean())},
        "per_step": [{k: r[k] for k in ("step", "loss", "grad_norm", "lr", "data_s", "compute_s")} for r in recs],
    }
    if dev.type == "cuda":
        out["peak_gpu_mem_allocated_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
        out["peak_gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 2**30
        try:
            out["gpu_utilization_pct_at_end"] = int(torch.cuda.utilization(dev))
        except Exception:                     # needs pynvml; optional
            out["gpu_utilization_pct_at_end"] = None
    return out


def run(cfg: dict, steps: int, out_dir: str | None = None, val_once: bool = False,
        allow_dirty: bool = False) -> dict:
    tc = TrainConfig.from_dict(cfg.get("train"))
    tcfg = TowerConfig.from_dict(cfg.get("model"))
    hcfg = F.HistoryConfig.from_dict(cfg.get("history"))
    if cfg.get("require_clean_git", False) and not allow_dirty:
        assert_clean_worktree()
    run_root = Path(os.environ.get("RUN_ROOT", cfg.get("run_root", str(paths.PROJECT_ROOT.parent / "kuairand_runs"))))
    out = Path(out_dir) if out_dir else run_root / "threetower" / f"bench_{cfg['exp']}" / time.strftime("%Y%m%d-%H%M%S")
    assert_outside_worktree(out)
    out.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    P = load_processed(_resolve(cfg.get("processed_dir", "data/processed")))
    cond = D.build_condition(cfg.get("condition", "normal"), P)
    user_vocab, user_table = user_features_for(cfg, cond)
    tr = Trainer(cond, tcfg, hcfg, tc, out, device, user_vocab, user_table)
    t_setup = time.time() - t0
    res = benchmark_steps(tr, steps)
    if val_once:
        # One validation pass, timed (D1-11 run-time estimate). Its score is not recorded as a metric.
        from kuairand_rec.models.evaluation import Validator
        ecfg = dict(cfg.get("eval", {}))
        t1 = time.time()
        v = Validator(P, cond, user_table, threads=ecfg.get("threads", 1),
                      user_batch_size=int(ecfg.get("val_user_batch_size", 256)), device=device)
        t_vsetup = time.time() - t1
        t2 = time.time()
        v(tr.model, 0)
        res["val_setup_s"], res["val_pass_s"] = t_vsetup, time.time() - t2
        res["run_upper_bound_h"] = tc.max_epochs * (res["epoch_s_estimate"] + res["val_pass_s"]) / 3600
    meta = {"note": NOTE, "exp": cfg["exp"], "git_sha": git_sha(), "config": cfg, "train_config": asdict(tc),
            "device": str(tr.device), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(tr.device) if tr.device.type == "cuda" else None,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "tf32": bool(torch.backends.cuda.matmul.allow_tf32), "setup_s": t_setup,
            "params": int(sum(p.numel() for p in tr.model.parameters())), "samples": tr.sample_counts,
            "condition_id": cond.condition_id, **res}
    (out / "benchmark.json").write_text(json.dumps(meta, indent=1, default=str))
    summary = {k: meta.get(k) for k in ("device", "gpu", "batch_size", "steps", "steps_per_s", "data_share",
                                        "epoch_s_estimate", "peak_gpu_mem_allocated_gb", "val_pass_s",
                                        "run_upper_bound_h")}
    summary.update(loss=meta["loss"], grad_norm=meta["grad_norm_pre_clip"], out=str(out / "benchmark.json"))
    print(json.dumps(summary, default=str), flush=True)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=None, help="benchmark override only")
    ap.add_argument("--device", default=None, help="override, e.g. cpu for a local dry run")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--val_once", action="store_true", help="also time one validation pass")
    ap.add_argument("--allow_dirty", action="store_true", help="local dry runs only; never for GPU runs")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.batch_size:
        cfg["train"] = {**cfg.get("train", {}), "batch_size": a.batch_size}
    if a.device:
        cfg["device"] = a.device
    run(cfg, a.steps, a.out_dir, a.val_once, a.allow_dirty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
