"""Check the extended report (report/REPORT_FULL.md, which carries every formal number of the
concise report/REPORT.md) against the formal result files (read-only on every input).

Run:  uv run python scripts/check_report_claims.py [--report report/REPORT.md]
Output: results/final_audit/report_claims.json (+ printed summary). Exit code 1 on any mismatch,
missing table, or missing text anchor.

What is checked automatically (the report text is actually parsed):
  1. Tables, located by their heading line in REPORT.md and parsed cell by cell:
     - "Table 1. Test R@20 / NDCG@20"     every cell (R@20 / N@20) vs bootstrap_models.csv
     - "Table 2. Paired differences"      every point estimate and its significance star (and the
                                          printed CI, where present) vs bootstrap_diffs.csv
     - "Table 3. Cold 3T-U0 minus ..."    cold model value, each difference, star / CI vs cold CSVs
     - "U0 − ALS (pure) by bucket"        every cell + star vs bootstrap_diffs.csv (bucket strata)
     - "Cold users" table                 every cell vs bootstrap_models.csv (cold frame)
     - §1.5 "Population" table            users / GT / seen-excluded / OOC / sentinel / cold / OOV
                                          vs frames_summary.json
     - "Table B1"                         every R/N@10/20/50 cell vs bootstrap_models.csv
     A table cell passes iff the formal value rounded to the printed number of decimals equals
     the printed value, and a printed star equals `ci_excludes_0`.
  2. Prose: a fixed list of key sentences / numbers (ablations, cold-video notes, error
     analysis E1-E5, validation numbers) is checked by locating an anchor string in REPORT.md
     and comparing the printed numbers that follow it with the formal value. If the anchor
     text is edited the check FAILS (it never silently passes).

Not checked automatically (manual review scope, listed in the JSON as `manual_review_scope`):
  qualitative statements and interpretations, numbers inside figure captions, dataset counts
  quoted from AUDIT.md / DATA_REPORT.md in §1.1-1.2, model sizes and hyperparameters in §1.3-1.4,
  and any number not covered by a table above or by the prose list below.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report/REPORT_FULL.md"   # override with --report
MAIN = ROOT / "results/m5/main_formal"
COLD = ROOT / "results/m5/cold_video/eval_671dde9_job33"

mo = pd.read_csv(MAIN / "bootstrap_models.csv")
di = pd.read_csv(MAIN / "bootstrap_diffs.csv")
cb = pd.read_csv(COLD / "cold_bootstrap_models.csv")
cd = pd.read_csv(COLD / "cold_bootstrap_diffs.csv")
fs = json.loads((ROOT / "results/baselines/frames_summary.json").read_text())
EA = json.loads((ROOT / "results/final_analysis/error_analysis.json").read_text())

TASK = {"video LV": "video_long_view", "video like": "video_is_like", "music LV": "music_long_view",
        "music like": "music_is_like"}
MODEL = {"Pop": "pop", "ALS (pure)": "als", "ALS pure": "als", "ALS (aux, like only)": "als_like_aux",
         "ALS (aux)": "als_like_aux", "ALS aux": "als_like_aux", "3T-U0 (main)": "u0", "3T-U0": "u0", "U0": "u0",
         "3T-U1 (+ snapshot)": "u1", "3T-U1": "u1", "U1": "u1", "no-history": "abl_nohist",
         "no-music": "abl_nomusic", "no-history ablation": "abl_nohist"}
NUM = re.compile(r"[+−-]?\d*\.\d+|[+−-]?\d[\d,]*")

RESULTS: list[dict] = []


def _num(s: str) -> float:
    return float(s.replace("−", "-").replace(",", ""))


def _dec(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def record(where: str, printed: str, value: float, sig_printed=None, sig_actual=None):
    ok = round(float(value), _dec(printed)) == round(_num(printed), _dec(printed))
    if sig_printed is not None:
        ok &= bool(sig_printed) == bool(sig_actual)
    RESULTS.append({"where": where, "printed": printed, "formal_value": float(value),
                    "sig_printed": sig_printed, "sig_formal": sig_actual, "ok": bool(ok)})


def fail(where: str, why: str):
    RESULTS.append({"where": where, "printed": None, "formal_value": None, "ok": False, "error": why})


def table_after(text: str, heading: str) -> list[list[str]]:
    """Rows (list of cells) of the first Markdown table after the line containing `heading`."""
    i = text.find(heading)
    if i < 0:
        raise KeyError(heading)
    rows = []
    for line in text[i:].splitlines()[1:]:
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif rows:
            break
    return rows


def m(task, model, metric="recall@20", stratum="all", kind="masked"):
    r = mo[(mo.task == task) & (mo.model == model) & (mo.metric == metric) & (mo.stratum == stratum)
           & (mo.frame_kind == kind)]
    return r.iloc[0] if len(r) == 1 else None


def d(task, a, b, metric="recall@20", stratum="all"):
    r = di[(di.task == task) & (di.model_a == a) & (di.model_b == b) & (di.metric == metric)
           & (di.stratum == stratum) & (di.frame_kind == "masked")]
    return r.iloc[0] if len(r) == 1 else None


def diff_cell(where, cell, row):
    """'+.032\\*' or '−.012\\* [−.016, −.008]' or '−.009 [−.025, +.005]' -> point, star, CI."""
    nums = NUM.findall(cell.replace("**", ""))
    star = "\\*" in cell or "*" in cell.replace("**", "")
    if row is None:
        return fail(where, "no formal row")
    record(where, nums[0], row["diff"], star, row["ci_excludes_0"])
    if len(nums) >= 3:
        record(where + " ci_lo", nums[1], row["ci_lo"])
        record(where + " ci_hi", nums[2], row["ci_hi"])


def check_tables(text: str):
    # Table 1
    rows = table_after(text, "**Table 1. Test R@20 / NDCG@20.**")
    hdr = rows[0]
    for r in rows[1:]:
        mod = MODEL[r[0]]
        for h, c in zip(hdr[1:], r[1:]):
            if c.strip("– ") == "":
                continue
            r20, n20 = [x.strip().strip("*") for x in c.split("/")]
            record(f"T1 {r[0]} {h} R@20", r20, m(TASK[h], mod)["mean"])
            record(f"T1 {r[0]} {h} N@20", n20, m(TASK[h], mod, "ndcg@20")["mean"])
    # Table 2
    rows = table_after(text, "**Table 2. Paired differences in R@20")
    hdr = rows[0]
    for r in rows[1:]:
        a, b = [MODEL[x.strip()] for x in r[0].split("−")]
        for h, c in zip(hdr[1:], r[1:]):
            if c.strip() == "–":
                continue
            diff_cell(f"T2 {r[0]} {h}", c, d(TASK[h], a, b))
    # Table 3 (cold video)
    rows = table_after(text, "**Table 3. Cold 3T-U0 minus each cold-condition baseline")
    for r in rows[1:]:
        task, frame = TASK[r[0]], {"full catalog": "cold_video_full", "H only": "cold_video_h"}[r[1]]
        cm = cb[(cb.task == task) & (cb.frame == frame) & (cb.model == "cold_threetower") & (cb.metric == "recall@20")]
        record(f"T3 {r[0]} {r[1]} cold3T", r[2], cm.iloc[0]["mean"])
        for col, bname in ((3, "cold_pop"), (4, "random_expected")):
            x = cd[(cd.task == task) & (cd.frame == frame) & (cd.model_a == "cold_threetower") & (cd.model_b == bname)
                   & (cd.metric == "recall@20")]
            diff_cell(f"T3 {r[0]} {r[1]} - {bname}", r[col], x.iloc[0] if len(x) == 1 else None)
    # Bucket table
    rows = table_after(text, "**U0 − ALS (pure) by bucket**")
    hdr = rows[0]
    for r in rows[1:]:
        for h, c in zip(hdr[1:], r[1:]):
            diff_cell(f"bucket {r[0]} {h}", c, d(TASK[r[0]], "u0", "als", stratum=h.replace("–", "-")))
    # Cold users
    rows = table_after(text, "**Cold users** (no history")
    for r in rows[1:]:
        record(f"cold users {r[0]}", r[1], m("video_long_view", MODEL[r[0]], kind="cold")["mean"])
    # §1.5 population table
    rows = table_after(text, "**Population.** A user is evaluated")
    keys = {"users": "n_users", "GT pairs": "gt_pairs_effective", "seen-excluded GT pairs": "gt_pairs_masked_seen",
            "OOC positive events": "pos_events_ooc", "sentinel-excluded events": "pos_events_sentinel",
            "train-OOV users": "evaluated_users_train_oov"}
    hdr = rows[0]
    for r in rows[1:]:
        f = fs[f"test/{TASK[r[0]]}/masked"]
        for h, c in zip(hdr[1:], r[1:]):
            if c == "–":
                record(f"pop {r[0]} {h}", "0", f[keys[h]] if h in keys else 0)
                continue
            if h == "cold users (separate)":
                record(f"pop {r[0]} {h}", c, fs[f"test/{TASK[r[0]]}/cold"]["n_users"])
            else:
                record(f"pop {r[0]} {h}", c, f[keys[h]])
    # Table B1
    rows = table_after(text, "**Table B1. Test metrics, all models.**")
    hdr = rows[0]
    for r in rows[1:]:
        for h, c in zip(hdr[2:], r[2:]):
            met = h.replace("R@", "recall@").replace("N@", "ndcg@")
            record(f"B1 {r[0]} {r[1]} {h}", c, m(TASK[r[0]], MODEL[r[1]], met)["mean"])


# Prose claims: (anchor text that must appear in REPORT.md, [(index of the number after the
# anchor, formal value, star or None)]).
def prose_specs():
    VL, VK, ML, MK = TASK["video LV"], TASK["video like"], TASK["music LV"], TASK["music like"]
    e1 = {(r["model_b"], g): r for g, rows in EA["E1"][VK].items() for r in rows}
    e2 = EA["E2"][VL]
    val = pd.read_csv(ROOT / "artifacts/u0/9218a76_seed0_job27/metrics.csv") if (
        ROOT / "artifacts/u0/9218a76_seed0_job27/metrics.csv").exists() else None
    bm = pd.read_csv(ROOT / "results/baselines/metrics.csv")

    def v(t, a):
        x = val[(val.split == "val") & (val.kind == "masked") & (val.target == t) & (val.action == a)]
        return x.iloc[0]["recall@20"]

    def b(t, a):
        x = bm[(bm.split == "val") & (bm.kind == "masked") & (bm.model == "als") & (bm.target == t) & (bm.action == a)]
        return x.iloc[0]["recall@20"]

    specs = [
        ("The NDCG@20 differences include 0 (", [(0, d(VL, "u0", "als", "ndcg@20"), False),
                                                (1, d(ML, "u0", "als", "ndcg@20"), False)]),
        ("U0's NDCG@10 is slightly *higher* on video LV (", [(0, d(VL, "u0", "als", "ndcg@10"), True)]),
        ("U0 is clearly behind at R@50 (", [(0, d(VL, "u0", "als", "recall@50"), True)]),
        ("**U0 vs Pop on like.** NDCG@20 is not significant (", [(0, d(VK, "u0", "pop", "ndcg@20"), False),
                                                                 (1, d(MK, "u0", "pop", "ndcg@20"), False)]),
        ("**U1 vs pure ALS on LV.** NDCG@20 is ", [(0, d(VL, "u1", "als", "ndcg@20"), True),
                                                  (0, d(ML, "u1", "als", "ndcg@20"), True)]),
        ("video-LV R@50 is lower (", [(0, d(VL, "u1", "als", "recall@50"), True)]),
        ("R@20 drops significantly on video LV (", [(0, d(VL, "abl_nohist", "u0"), True),
                                                    (1, d(ML, "abl_nohist", "u0"), True),
                                                    (2, d(MK, "abl_nohist", "u0"), True)]),
        ("NDCG@20 on LV drops by ", [(0, d(VL, "abl_nohist", "u0", "ndcg@20"), True),
                                     (1, d(ML, "abl_nohist", "u0", "ndcg@20"), True)]),
        ("LV *improves*: R@20 ", [(0, d(VL, "abl_nomusic", "u0"), True), (1, d(ML, "abl_nomusic", "u0"), True),
                                  (2, d(VL, "abl_nomusic", "u0", "ndcg@20"), True),
                                  (3, d(ML, "abl_nomusic", "u0", "ndcg@20"), True)]),
        ("NDCG@10 is even slightly *below* random (", [
            (0, cd[(cd.task == VL) & (cd.frame == "cold_video_full") & (cd.model_a == "cold_threetower")
                   & (cd.model_b == "random_expected") & (cd.metric == "ndcg@10")].iloc[0], True)]),
        ("while R@50 is above it (", [
            (0, cd[(cd.task == VL) & (cd.frame == "cold_video_full") & (cd.model_a == "cold_threetower")
                   & (cd.model_b == "random_expected") & (cd.metric == "recall@50")].iloc[0], True)]),
        ("On those users U0 reaches R@20 ", [(0, {"diff": e1[("als", "als_no_signal")]["mean_a"]}, None),
                                             (1, e1[("als", "als_no_signal")], True),
                                             (2, e1[("als_like_aux", "als_no_signal")], False)]),
        ("On users that do have a prior like, U0 − pure ALS is ", [(0, e1[("als", "als_signal")], False)]),
        ("Also on those users, U0 is *below* ALS aux: ", [(0, e1[("als_like_aux", "als_signal")], True)]),
        ("U0 − ALS on video LV is ", [(0, e2["21-100 (input not truncated)"][0], True),
                                      (1, e2[">100 (input truncated to last 100)"][0], True)]),
        ("correlate only weakly with train LV popularity (Spearman ", [
            (0, {"diff": EA["E3"]["u0"]["spearman_mean_cold_score_vs_train_lv_popularity"]}, None)]),
        ("The no-history ablation's embedding correlates at ", [
            (0, {"diff": EA["E3"]["abl_nohist"]["spearman_mean_cold_score_vs_train_lv_popularity"]}, None)]),
        ("but the cold model gives H only ", [
            (0, {"diff": 100 * EA["E4"][VL]["cold_threetower"]["mean_share_of_top20_slots_on_H"]}, None)]),
        ("The best GT_H item has median rank ", [
            ("int", {"diff": EA["E4"][VL]["cold_threetower"]["median_rank_of_best_gt_h_item"]}, None)]),
        ("correlates at r = ", [(0, {"diff": EA["E5"]["pearson_r_u0_recall20_video_lv_vs_music_lv"]}, None)]),
        ("The normally trained U0 scores R@20 ", [
            (0, cb[(cb.task == VL) & (cb.frame == "cold_video_full") & (cb.model == "seen_threetower")
                   & (cb.metric == "recall@20")].iloc[0].rename({"mean": "diff"}), None),
            (1, cb[(cb.task == VL) & (cb.frame == "cold_video_h") & (cb.model == "seen_threetower")
                   & (cb.metric == "recall@20")].iloc[0].rename({"mean": "diff"}), None)]),
    ]
    if val is not None:
        specs.append(("U0 was below ALS on validation LV R@20 (video ", [
            (0, {"diff": v("video", "long_view")}, None), (1, {"diff": b("video", "long_view")}, None),
            (2, {"diff": v("music", "long_view")}, None), (3, {"diff": b("music", "long_view")}, None)]))
    return specs


def check_prose(text: str):
    for anchor, items in prose_specs():
        i = text.find(anchor)
        if i < 0:
            fail(f"prose: {anchor[:50]}", "anchor text not found in REPORT.md")
            continue
        seg = text[i + len(anchor): i + len(anchor) + 220]
        toks = list(re.finditer(r"([+−-]?\d*\.\d+)(\\\*)?", seg))
        for k, row, star in items:
            if k == "int":            # a single integer right after the anchor (e.g. a rank)
                mi = re.match(r"\d+", seg)
                if not mi:
                    fail(f"prose: {anchor[:50]}", "no integer after the anchor")
                    continue
                RESULTS.append({"where": f"prose: {anchor[:50]}", "printed": mi.group(0),
                                "formal_value": float(row["diff"]),
                                "ok": abs(float(mi.group(0)) - float(row["diff"])) <= 0.5})
                continue
            if k >= len(toks):
                fail(f"prose: {anchor[:50]} #{k}", "fewer numbers than expected after the anchor")
                continue
            t = toks[k]
            val = row["diff"]
            record(f"prose: {anchor[:50]} #{k}", t.group(1), val,
                   None if star is None else bool(t.group(2)), None if star is None else bool(row["ci_excludes_0"]))


MANUAL_SCOPE = [
    "qualitative statements, interpretations and hypotheses (Error analysis, Limitations, Next)",
    "numbers in figure captions (figures themselves are generated from the formal CSVs)",
    "dataset counts in §1.1-1.2 (cross-checked by hand against results/audit/AUDIT.md, "
    "results/data_pipeline/DATA_REPORT.md and results/final_audit/checks.json)",
    "model sizes, hyperparameters, sample counts in §1.3-1.4 (run_meta.json / configs)",
    "abstract numbers (they repeat Table 2 / Table 3 values that are checked automatically)",
]


def main() -> int:
    global REPORT
    if "--report" in sys.argv:
        REPORT = (ROOT / sys.argv[sys.argv.index("--report") + 1]).resolve()
    text = REPORT.read_text()
    try:
        check_tables(text)
    except KeyError as e:
        fail("tables", f"table heading not found: {e}")
    check_prose(text)
    bad = [r for r in RESULTS if not r["ok"]]
    out = ROOT / "results/final_audit/report_claims.json"
    out.write_text(json.dumps({
        "report": str(REPORT.relative_to(ROOT)), "method": "parses REPORT.md tables + anchored prose numbers",
        "n_checked": len(RESULTS), "n_failed": len(bad),
        "n_table_cells": sum(not r["where"].startswith("prose") for r in RESULTS),
        "n_prose_numbers": sum(r["where"].startswith("prose") for r in RESULTS),
        "manual_review_scope": MANUAL_SCOPE, "checks": RESULTS}, indent=1, default=float))
    for r in bad:
        print("MISMATCH", r)
    print(json.dumps({"n_checked": len(RESULTS), "n_failed": len(bad),
                      "table_cells": sum(not r["where"].startswith("prose") for r in RESULTS),
                      "prose_numbers": sum(r["where"].startswith("prose") for r in RESULTS)}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
