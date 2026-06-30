#!/usr/bin/env python3
"""
Extract per-condition test results from CLFTv2 log directories and print
tables ready to paste into the paper.

Usage:
    python extract_results.py                           # all groups, ablation table
    python extract_results.py --group shared llava      # specific groups
    python extract_results.py --table weather           # weather breakdown
    python extract_results.py --table modality          # modality ablation
    python extract_results.py --table agent_performance # agent signal agreement
    python extract_results.py --table all               # every table
    python extract_results.py --format csv              # CSV output
"""

import argparse
import json
from pathlib import Path

LOGS_ROOT = Path("logs/vlm/clftv2-base")

# VLM pipeline results (frame_*.json written by process_frames.py)
VLM_RESULTS_ROOT = Path("/run/media/tom/ml/zod_temp/vlm")
VLM_TAGS = {"llava": "llava_34b", "qwen": "qwen2.5vl_72b"}

CONDITIONS = ["day_fair", "day_rain", "night_fair", "night_rain", "snow"]
CLASSES = ["vehicle", "sign", "human"]
METRICS = ["mIoU", "vehicle", "sign", "human", "fw_iou"]

COND_LABELS = {
    "day_fair":   "Day-fair",
    "day_rain":   "Day-rain",
    "night_fair": "Night-fair",
    "night_rain": "Night-rain",
    "snow":       "Snow",
    "overall":    "Overall",
}

# Maps logdir variant name → paper row label (in display order)
VARIANT_LABELS: dict[str, str] = {
    # Shared / VLM-independent
    "raw_sam_fusion":               "Raw SAM",
    "swin_only_fusion":             "Swin only",
    "swin_discovery_noVLM_fusion":  "Swin + disc. (no VLM)",
    "gt_fusion":                    "GT (CrossFusion)",
    "gt_rgb":                       "GT (RGB)",
    "gt_lidar":                     "GT (LiDAR)",
    # VLM-dependent — main variants
    "vlm_only_fusion":              "VLM only",
    "triage_fusion":                "Triage",
    "raw_sam_discovery_fusion":     "Raw SAM + disc.",
    "swin_discovery_fusion":        "Swin + disc.",
    "full_fusion":                  "Triage + disc.",
    # Modality ablation
    "swin_discovery_rgb":           "Swin + disc. (RGB)",
    "swin_discovery_lidar":         "Swin + disc. (LiDAR)",
    # Triage rule ablations
    "disjunctive_reject_fusion":    "Disjunctive reject",
    "uniform_tau_fusion":           "Uniform tau_q",
}

# Variants included in each paper table (in order)
ABLATION_VARIANTS = [
    "raw_sam_fusion",
    "vlm_only_fusion",
    "triage_fusion",
    "swin_only_fusion",
    "raw_sam_discovery_fusion",
    "swin_discovery_fusion",
    "swin_discovery_noVLM_fusion",
    "full_fusion",
    "disjunctive_reject_fusion",
    "uniform_tau_fusion",
]

WEATHER_VARIANTS = [
    "raw_sam_fusion",
    "swin_discovery_noVLM_fusion",
    "swin_discovery_fusion",
    "full_fusion",
]

MODALITY_VARIANTS = [
    "gt_rgb",
    "gt_lidar",
    "gt_fusion",
    "swin_discovery_rgb",
    "swin_discovery_lidar",
    "swin_discovery_fusion",
]


# ── Result loading ─────────────────────────────────────────────────────────────

def load_test_result(variant_dir: Path) -> dict | None:
    """Return parsed JSON from the single test_results file in variant_dir."""
    results_dir = variant_dir / "test_results"
    if not results_dir.exists():
        return None
    files = sorted(results_dir.glob("*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


def collect_results(groups: list[str]) -> dict[str, dict[str, dict]]:
    """
    Returns nested dict: group → variant_name → parsed test result data.
    Skips variants with no test_results file.
    """
    out: dict[str, dict[str, dict]] = {}
    for group in groups:
        group_dir = LOGS_ROOT / group
        if not group_dir.exists():
            continue
        out[group] = {}
        for variant_dir in sorted(group_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            result = load_test_result(variant_dir)
            if result:
                out[group][variant_dir.name] = result
    return out


# ── Metric extraction ──────────────────────────────────────────────────────────

def _extract_classes_overall(result: dict) -> dict:
    """
    Per-class IoU averaged across all weather conditions.
    Mirrors how overall mIoU is computed across the test set.
    """
    cond_results = [result["test_results"][c] for c in CONDITIONS
                    if c in result["test_results"]]
    out = {}
    for cls in CLASSES:
        vals = [c[cls]["iou"] for c in cond_results if cls in c]
        out[cls] = sum(vals) / len(vals) if vals else float("nan")
    return out


def _fw_iou_from_aggregated_cm(result: dict) -> float:
    """
    Compute fw-IoU from the sum of per-condition confusion matrices.

    The stored overall.fw_iou is a macro-average of per-condition values, which
    is incorrect for a frequency-weighted metric: different conditions have different
    frame counts, so averaging discards the global pixel distribution.
    Summing the confusion matrices first gives the correct global weighting.
    Background (class 0) is excluded to match per-condition fw_iou computation.
    """
    import numpy as np
    cm = None
    for cond in CONDITIONS:
        cond_cm = (result["test_results"]
                   .get(cond, {})
                   .get("overall", {})
                   .get("confusion_matrix"))
        if cond_cm is None:
            continue
        cond_cm = np.array(cond_cm, dtype=float)
        cm = cond_cm if cm is None else cm + cond_cm
    if cm is None:
        return float("nan")
    n = cm.shape[0]
    ious, freqs = [], []
    for i in range(1, n):  # skip background (class 0)
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        ious.append(float(tp / denom) if denom > 0 else 0.0)
        freqs.append(float(cm[i, :].sum()))
    total_fg = sum(freqs)
    if total_fg == 0:
        return float("nan")
    return sum(freqs[i] / total_fg * ious[i] for i in range(len(ious)))


def get_overall_metrics(result: dict) -> dict:
    """Return overall mIoU, per-class IoU, and fw_iou for a result."""
    overall = result["test_results"].get("overall", {})
    class_avg = _extract_classes_overall(result)
    return {
        **class_avg,
        "mIoU":   overall.get("mIoU_foreground", float("nan")),
        "fw_iou": _fw_iou_from_aggregated_cm(result),
    }


def get_per_condition(result: dict) -> dict[str, dict]:
    """
    Extract all metrics per weather condition: per-class IoU, mIoU, and fw_iou.
    Also includes an 'overall' entry aggregated across conditions.
    """
    out = {}
    for cond in CONDITIONS:
        if cond not in result["test_results"]:
            continue
        c = result["test_results"][cond]
        out[cond] = {
            "mIoU":   c["overall"]["mIoU_foreground"],
            "fw_iou": c["overall"]["fw_iou"],
            **{cls: c[cls]["iou"] for cls in CLASSES if cls in c},
        }
    overall_raw = result["test_results"].get("overall", {})
    class_avg = _extract_classes_overall(result)
    out["overall"] = {
        "mIoU":   overall_raw.get("mIoU_foreground", float("nan")),
        "fw_iou": overall_raw.get("fw_iou", float("nan")),
        **class_avg,
    }
    return out


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(v: float | None) -> str:
    if v is None or v != v:  # nan check
        return "   --"
    return f"{v * 100:5.1f}"


def _label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def _group_tag(group: str, variant: str = "") -> str:
    if group == "shared" and variant.startswith("gt_"):
        return "GT"
    tags = {"llava": "LLaVA-1.6-34B", "qwen": "Qwen2.5-VL-72B", "shared": "shared"}
    return tags.get(group, group)


# ── Table printers ─────────────────────────────────────────────────────────────

def print_ablation(results: dict, fmt_csv: bool = False) -> None:
    groups = [g for g in ("shared", "llava", "qwen") if g in results]
    header_groups = [g for g in groups if g != "shared"]

    sep = "," if fmt_csv else "  "
    col_w = 24
    # Columns per group: Veh, Sign, Human, mIoU, fw_iou
    cols = ["Veh", "Sign", "Human", "mIoU", "fw_iou"]
    n_cols = len(cols)
    col_block = n_cols * 7  # each value is 7 chars wide

    if not fmt_csv:
        print("\n── Ablation table (overall) ──────────────────────────────────────────────")
        group_headers = "".join(
            f"  {'── ' + _group_tag(g) + ' ──':^{col_block}}" for g in header_groups
        )
        print(f"{'Variant':<{col_w}}{group_headers}")
        print(f"{'':^{col_w}}" + "".join(
            f"  {'Veh':>6}{'Sign':>7}{'Human':>7}{'mIoU':>7}{'fw_iou':>8}" for _ in header_groups
        ))
        print("-" * (col_w + (col_block + 2) * len(header_groups)))
    else:
        print(sep.join(
            ["Variant"] + [f"{_group_tag(g)}_{c}" for g in header_groups for c in cols]
        ))

    def _row_vals(m: dict) -> list[str]:
        return [fmt(m[c]) for c in CLASSES] + [fmt(m["mIoU"]), fmt(m["fw_iou"])]

    # Shared rows (VLM-independent — same value in every group column)
    shared = results.get("shared", {})
    for variant in ABLATION_VARIANTS:
        if variant not in shared:
            continue
        m = get_overall_metrics(shared[variant])
        vals = _row_vals(m)
        if fmt_csv:
            print(sep.join([_label(variant)] + vals * len(header_groups)))
        else:
            row = "".join(f"  {''.join(f'{v:>6}' for v in vals[:3])}{vals[3]:>7}{vals[4]:>8}")
            print(f"{_label(variant):<{col_w}}" + row * len(header_groups))

    # VLM-dependent rows
    if not fmt_csv:
        print()
    for variant in ABLATION_VARIANTS:
        if variant in shared:
            continue
        row_cells = []
        found_any = False
        for group in header_groups:
            gdata = results.get(group, {})
            if variant in gdata:
                m = get_overall_metrics(gdata[variant])
                row_cells.append(_row_vals(m))
                found_any = True
            else:
                row_cells.append(["   --"] * n_cols)
        if not found_any:
            continue
        if fmt_csv:
            print(sep.join([_label(variant)] + [v for cell in row_cells for v in cell]))
        else:
            row = ""
            for cell in row_cells:
                row += f"  {''.join(f'{v:>6}' for v in cell[:3])}{cell[3]:>7}{cell[4]:>8}"
            print(f"{_label(variant):<{col_w}}{row}")


def print_weather(results: dict, fmt_csv: bool = False) -> None:
    all_results = {**results.get("shared", {}), **results.get("llava", {}), **results.get("qwen", {})}
    sep = "," if fmt_csv else "  "
    conds_with_overall = CONDITIONS + ["overall"]
    col_w = 24
    metric_w = 10

    if fmt_csv:
        cond_metric_cols = [f"{COND_LABELS[c]}_{m}" for c in conds_with_overall for m in METRICS]
        print(sep.join(["Variant"] + cond_metric_cols))
        for variant in WEATHER_VARIANTS:
            if variant not in all_results:
                continue
            per_cond = get_per_condition(all_results[variant])
            vals = [fmt(per_cond.get(c, {}).get(m)) for c in conds_with_overall for m in METRICS]
            print(sep.join([_label(variant)] + vals))
        return

    cond_header = "".join(f"  {COND_LABELS[c]:>{metric_w}}" for c in conds_with_overall)
    separator = "-" * (col_w + 14 + (metric_w + 2) * len(conds_with_overall))

    print("\n── Weather breakdown (per-class IoU + fw_iou) ────────────────────────────")
    print(f"{'Variant':<{col_w}}  {'Metric':<10}" + cond_header)
    print(separator)

    for variant in WEATHER_VARIANTS:
        if variant not in all_results:
            continue
        per_cond = get_per_condition(all_results[variant])
        first = True
        for metric in METRICS:
            label = _label(variant) if first else ""
            first = False
            vals = "".join(
                f"  {fmt(per_cond.get(c, {}).get(metric)):>{metric_w}}"
                for c in conds_with_overall
            )
            print(f"{label:<{col_w}}  {metric:<10}{vals}")
        print()


def print_modality(results: dict, fmt_csv: bool = False) -> None:
    sep = "," if fmt_csv else "  "
    col_w = 26

    if not fmt_csv:
        print("\n── Modality ablation ─────────────────────────────────────────────────────")
        print(f"{'Variant':<{col_w}}  {'Group':>14}  {'Veh':>6}{'Sign':>7}{'Human':>7}{'mIoU':>7}{'fw_iou':>8}")
        print("-" * (col_w + 52))
    else:
        print(sep.join(["Variant", "Group", "Veh", "Sign", "Human", "mIoU", "fw_iou"]))

    for variant in MODALITY_VARIANTS:
        for group in ("shared", "llava", "qwen"):
            gdata = results.get(group, {})
            if variant not in gdata:
                continue
            m = get_overall_metrics(gdata[variant])
            vals = [fmt(m[c]) for c in CLASSES] + [fmt(m["mIoU"]), fmt(m["fw_iou"])]
            tag = _group_tag(group, variant)
            if fmt_csv:
                print(sep.join([_label(variant), tag] + vals))
            else:
                print(f"{_label(variant):<{col_w}}  {tag:>14}  " +
                      "".join(f"{v:>6}" for v in vals[:3]) +
                      f"{vals[3]:>7}{vals[4]:>8}")


# ── Agent performance ──────────────────────────────────────────────────────────

def _load_agent_records(tag: str) -> dict[str, dict]:
    """Load all frame_*.json files for a VLM run tag. Returns {frame_id: data}."""
    results_dir = VLM_RESULTS_ROOT / tag / "results"
    if not results_dir.exists():
        return {}
    out = {}
    for f in sorted(results_dir.glob("frame_*.json")):
        data = json.loads(f.read_text())
        out[data["frame_id"]] = data
    return out


def _pr(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    acc  = (tp + tn) / n          if n           else float("nan")
    prec = tp / (tp + fp)         if (tp + fp)   else float("nan")
    rec  = tp / (tp + fn)         if (tp + fn)   else float("nan")
    f1   = 2*prec*rec/(prec+rec)  if (prec+rec)  else float("nan")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "n": n}


def compute_agent_performance() -> dict:
    """
    Compute agent signal agreement statistics using Swin quality as the reference.

    Reference positive = swin_agreement >= per-class swin_q_threshold (stored per mask).
    Swin scores and LiDAR consistency are VLM-independent, so values from the LLaVA
    run are used as the shared reference.  BBox verdicts differ per VLM.
    """
    llava = _load_agent_records(VLM_TAGS["llava"])
    qwen  = _load_agent_records(VLM_TAGS["qwen"])

    if not llava:
        return {}

    counters: dict[str, dict] = {
        "bbox_llava":   {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "bbox_qwen":    {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "consistency":  {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
    }
    cross = {"agree": 0, "total": 0}

    common_frames = set(llava) & set(qwen)
    for fid in sorted(common_frames):
        l_masks = {m["mask_id"]: m for m in llava[fid]["masks"]}
        q_masks = {m["mask_id"]: m for m in qwen[fid]["masks"]}

        for mid in set(l_masks) & set(q_masks):
            lm = l_masks[mid]
            qm = q_masks[mid]

            sc = lm.get("scores", {})
            swin_score = sc.get("swin_agreement")
            if swin_score is None:
                continue
            swin_ref = swin_score >= sc.get("swin_q_threshold", 0.30)

            def _tally(key: str, pred_pos: bool) -> None:
                c = counters[key]
                if swin_ref and pred_pos:       c["tp"] += 1
                elif not swin_ref and pred_pos: c["fp"] += 1
                elif swin_ref:                  c["fn"] += 1
                else:                           c["tn"] += 1

            bbox_l = lm["agents"].get("bbox")
            if bbox_l is not None:
                _tally("bbox_llava", bbox_l == "valid")

            bbox_q = qm["agents"].get("bbox")
            if bbox_q is not None:
                _tally("bbox_qwen", bbox_q == "valid")

            cons = lm["agents"].get("consistency")
            if cons is not None:
                _tally("consistency", cons == "pass")

            if bbox_l is not None and bbox_q is not None:
                cross["total"] += 1
                if (bbox_l == "valid") == (bbox_q == "valid"):
                    cross["agree"] += 1

    cvl = cross["agree"] / cross["total"] if cross["total"] else float("nan")
    return {
        "metrics": {k: _pr(**v) for k, v in counters.items()},
        "cross_vlm_agreement": cvl,
        "cross_vlm_total": cross["total"],
        "n_frames": len(common_frames),
    }


def print_agent_performance(fmt_csv: bool = False) -> None:
    stats = compute_agent_performance()
    if not stats:
        print("# No agent results found — check VLM_RESULTS_ROOT paths")
        return

    m   = stats["metrics"]
    cvl = stats["cross_vlm_agreement"]

    AGENT_LABELS = {
        "bbox_llava":  "BBox VLM (LLaVA-1.6-34B)",
        "bbox_qwen":   "BBox VLM (Qwen2.5-VL-72B)",
        "consistency": "LiDAR consistency (det.)",
    }

    def fv(v: float) -> str:
        return "  --" if v != v else f"{v*100:5.1f}"

    if fmt_csv:
        print("Agent,Acc,Prec,Rec,F1,N")
        for key, label in AGENT_LABELS.items():
            d = m[key]
            print(f"{label},{fv(d['acc'])},{fv(d['prec'])},{fv(d['rec'])},{fv(d['f1'])},{d['n']}")
        print(f"Cross-VLM BBox agreement,{fv(cvl)},,,{stats['cross_vlm_total']}")
        return

    col_w = 30
    print("\n── Agent signal agreement (vs Swin quality reference) ────────────────────")
    print(f"  Reference: Swin quality 'good' = swin_agreement ≥ per-class τ_q")
    print(f"  Frames: {stats['n_frames']:,}  |  Positive = mask passes Swin quality check")
    print()
    print(f"  {'Agent':<{col_w}}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>8}")
    print("  " + "-" * (col_w + 38))
    for key, label in AGENT_LABELS.items():
        d = m[key]
        print(f"  {label:<{col_w}}  {fv(d['acc']):>6}  {fv(d['prec']):>6}"
              f"  {fv(d['rec']):>6}  {fv(d['f1']):>6}  {d['n']:>8,}")
    print()
    print(f"  Cross-VLM BBox agreement: {fv(cvl)}%  (n={stats['cross_vlm_total']:,})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLFTv2 test results for paper tables.")
    parser.add_argument("--group", nargs="+", default=["shared", "llava", "qwen"],
                        help="Log groups to include (default: shared llava qwen)")
    parser.add_argument("--table", default="ablation",
                        choices=["ablation", "weather", "modality", "agent_performance", "all"],
                        help="Which table to print (default: ablation)")
    parser.add_argument("--format", dest="fmt", default="table",
                        choices=["table", "csv"],
                        help="Output format (default: table)")
    args = parser.parse_args()

    results = collect_results(args.group)

    missing = [g for g in args.group if g not in results]
    if missing:
        print(f"# Note: no results found for group(s): {', '.join(missing)}")

    csv = args.fmt == "csv"
    if args.table in ("ablation", "all"):
        print_ablation(results, csv)
    if args.table in ("weather", "all"):
        print_weather(results, csv)
    if args.table in ("modality", "all"):
        print_modality(results, csv)
    if args.table in ("agent_performance", "all"):
        print_agent_performance(csv)


if __name__ == "__main__":
    main()
