#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count pixels per class for each dataset (Waymo, ZOD, ISEAuto).

Uses class relabeling from config JSON files so counts match training class indices.
Outputs LaTeX tables (train/val/test with per-weather-condition breakdown)
into paper/tables/ as {dataset}_pixels.tex.
"""

import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_CONFIGS = {
    "waymo": {
        "config_json": os.path.join(REPO_ROOT, "config/waymo/swin/config_1.json"),
        "dataset_root": os.path.join(REPO_ROOT, "waymo_dataset"),
        "splits": {
            "train": ["splits_clft/train_all.txt"],
            "val":   ["splits_clft/early_stop_valid.txt"],
            "test":  {
                "Day Fair":   "splits_clft/test_day_fair.txt",
                "Day Rain":   "splits_clft/test_day_rain.txt",
                "Night Fair": "splits_clft/test_night_fair.txt",
                "Night Rain": "splits_clft/test_night_rain.txt",
                "Snow":       "splits_clft/test_snow.txt",
            },
        },
        "annotation_fn": lambda p: p.replace("camera/", "annotation/"),
        "weather_from_path": lambda p: _waymo_weather(p),
        "resolution": "384×384",
        "label": "Waymo",
    },
    "zod": {
        "config_json": os.path.join(REPO_ROOT, "config/zod/swin/config_1.json"),
        "dataset_root": os.path.join(REPO_ROOT, "zod_dataset"),
        "frame_analysis_json": os.path.join(REPO_ROOT, "scripts/frame_analysis.json"),
        "splits": {
            "train": ["train.txt"],
            "val":   ["validation.txt"],
            "test":  {
                "Day Fair":   "test_day_fair.txt",
                "Day Rain":   "test_day_rain.txt",
                "Night Fair": "test_night_fair.txt",
                "Night Rain": "test_night_rain.txt",
                "Snow":       "test_snow.txt",
            },
        },
        "annotation_fn": lambda p, cfg=None: p.replace("camera", "annotation_camera_only"),
        "weather_from_path": None,   # ZOD train/val use frame_analysis.json for weather
        "resolution": "384×384",
        "label": "ZOD",
    },
    "iseauto": {
        "config_json": os.path.join(REPO_ROOT, "config/iseauto/swin/config_1.json"),
        "dataset_root": os.path.join(REPO_ROOT, "xod_dataset"),
        "splits": {
            "train": ["train.txt"],
            "val":   ["validation.txt"],
            "test":  {
                "Day Fair":   "test_day_fair.txt",
                "Day Rain":   "test_day_rain.txt",
                "Night Fair": "test_night_fair.txt",
                "Night Rain": "test_night_rain.txt",
                "Snow":       "test_snow.txt",
            },
        },
        "annotation_fn": lambda p: p.replace("camera", "annotation"),
        "weather_from_path": lambda p: _iseauto_weather(p),
        "resolution": "384×384",
        "label": "ISEAuto",
    },
}

OUTPUT_DIR = os.path.join(REPO_ROOT, "paper/tables")

# ---------------------------------------------------------------------------
# Weather detection helpers
# ---------------------------------------------------------------------------

def _waymo_weather(path: str) -> str:
    """Extract weather label from Waymo path structure: labeled/{day|night}/{not_rain|rain}/..."""
    parts = path.replace("\\", "/").split("/")
    try:
        labeled_idx = parts.index("labeled")
        time  = parts[labeled_idx + 1]   # day / night
        rain  = parts[labeled_idx + 2]   # not_rain / rain
    except (ValueError, IndexError):
        return "Unknown"
    time_str  = "Day"   if time  == "day"   else "Night"
    rain_str  = "Rain"  if rain  == "rain"  else "Fair"
    return f"{time_str} {rain_str}"


def _iseauto_weather(path: str) -> str:
    """Extract weather label from ISEAuto path: labeled/{day_fair|day_rain|night_fair|night_rain}/..."""
    parts = path.replace("\\", "/").split("/")
    mapping = {
        "day_fair":   "Day Fair",
        "day_rain":   "Day Rain",
        "night_fair": "Night Fair",
        "night_rain": "Night Rain",
    }
    for part in parts:
        if part in mapping:
            return mapping[part]
    return "Unknown"

# ---------------------------------------------------------------------------
# Class mapping from config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_relabel_table(config: dict):
    """Build a numpy lookup array: dataset_index → train_index."""
    train_classes = config["Dataset"]["train_classes"]
    max_ds_idx = max(
        idx
        for cls in train_classes
        for idx in cls["dataset_mapping"]
    )
    table = np.zeros(max_ds_idx + 2, dtype=np.int32)  # +2 for safety
    for cls in train_classes:
        for ds_idx in cls["dataset_mapping"]:
            table[ds_idx] = cls["index"]
    return table


def get_train_class_names(config: dict) -> list:
    """Return train class names sorted by training index."""
    return [c["name"] for c in sorted(config["Dataset"]["train_classes"], key=lambda c: c["index"])]

# ---------------------------------------------------------------------------
# Pixel counting
# ---------------------------------------------------------------------------

TARGET_RESOLUTION = (384, 384)  # (width, height) — matches training resize


def count_pixels_in_file(anno_path: str, relabel_table: np.ndarray, n_classes: int) -> np.ndarray:
    """Load annotation PNG, resize to TARGET_RESOLUTION, return per-class pixel counts."""
    img = Image.open(anno_path).convert("L")
    if img.size != TARGET_RESOLUTION:
        img = img.resize(TARGET_RESOLUTION, Image.NEAREST)
    arr = np.array(img, dtype=np.int32)
    # Clamp to valid range before lookup
    arr = np.clip(arr, 0, len(relabel_table) - 1)
    relabeled = relabel_table[arr]
    counts = np.zeros(n_classes, dtype=np.int64)
    for ci in range(n_classes):
        counts[ci] = np.sum(relabeled == ci)
    return counts


_WEATHER_LABEL_MAP = {
    "day_fair":   "Day Fair",
    "day_rain":   "Day Rain",
    "night_fair": "Night Fair",
    "night_rain": "Night Rain",
    "snow":       "Snow",
}


def load_frame_analysis(json_path: str) -> dict:
    """Load frame_analysis.json and return a dict keyed by frame_id string."""
    with open(json_path) as f:
        data = json.load(f)
    return data["frames"]


def _frame_id_from_path(cam_path: str) -> str:
    """Extract 6-digit frame id from a path like 'camera/frame_000004.png'."""
    basename = os.path.splitext(os.path.basename(cam_path))[0]  # 'frame_000004'
    return basename.split("_", 1)[1] if "_" in basename else basename  # '000004'


def process_split_file_from_json(
    split_txt: str,
    frames: dict,
    relabel_table: np.ndarray,
    n_classes: int,
) -> dict:
    """
    Count pixels per class per weather using pre-computed frame_analysis.json.

    Returns dict: weather_label → {"samples": int, "pixels": np.ndarray[n_classes]}
    """
    if not os.path.exists(split_txt):
        print(f"  [SKIP] {split_txt} not found")
        return {}

    with open(split_txt) as f:
        lines = [l.strip() for l in f if l.strip()]

    results = defaultdict(lambda: {"samples": 0, "pixels": np.zeros(n_classes, dtype=np.int64)})

    missing = 0
    for rel_cam_path in lines:
        frame_id = _frame_id_from_path(rel_cam_path)
        if frame_id not in frames:
            missing += 1
            continue

        frame = frames[frame_id]
        weather = _WEATHER_LABEL_MAP.get(frame.get("weather", ""), "Unknown")

        # pixel_counts keys are raw dataset indices (strings) at original resolution
        raw_counts = frame["pixel_counts"]
        orig_total  = int(frame["total_pixels"])
        target_total = TARGET_RESOLUTION[0] * TARGET_RESOLUTION[1]  # 384*384
        scale = target_total / orig_total if orig_total > 0 else 1.0

        max_idx = max(int(k) for k in raw_counts)
        raw_arr = np.zeros(max(max_idx + 1, len(relabel_table)), dtype=np.float64)
        for k, v in raw_counts.items():
            idx = int(k)
            if idx < len(raw_arr):
                raw_arr[idx] = float(v) * scale

        # Apply relabeling: accumulate scaled pixels into training class bins
        counts = np.zeros(n_classes, dtype=np.int64)
        for ds_idx in range(min(len(relabel_table), len(raw_arr))):
            train_idx = int(relabel_table[ds_idx])
            if train_idx < n_classes:
                counts[train_idx] += int(round(raw_arr[ds_idx]))

        results[weather]["samples"] += 1
        results[weather]["pixels"]  += counts

    if missing:
        print(f"  [WARN] {missing} frames not found in frame_analysis.json")

    return dict(results)


def process_split_file(
    split_txt: str,
    dataset_root: str,
    annotation_fn,
    relabel_table: np.ndarray,
    n_classes: int,
    weather_fn=None,
) -> dict:
    """
    Read a split .txt file and count pixels per class, optionally grouped by weather.

    Returns dict: weather_label → {"samples": int, "pixels": np.ndarray[n_classes]}
    If weather_fn is None, all go under key "All".
    """
    if not os.path.exists(split_txt):
        print(f"  [SKIP] {split_txt} not found")
        return {}

    with open(split_txt) as f:
        lines = [l.strip() for l in f if l.strip()]

    results = defaultdict(lambda: {"samples": 0, "pixels": np.zeros(n_classes, dtype=np.int64)})

    total = len(lines)
    for i, rel_cam_path in enumerate(lines):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{total} ...", flush=True)

        weather = weather_fn(rel_cam_path) if weather_fn else "All"
        rel_anno_path = annotation_fn(rel_cam_path)
        anno_path = os.path.join(dataset_root, rel_anno_path)

        if not os.path.exists(anno_path):
            print(f"  [WARN] missing annotation: {anno_path}")
            continue

        counts = count_pixels_in_file(anno_path, relabel_table, n_classes)
        results[weather]["samples"] += 1
        results[weather]["pixels"] += counts

    return dict(results)

# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

WEATHER_ORDER = ["Day Fair", "Day Rain", "Night Fair", "Night Rain", "Snow", "All"]


def _fmt_count(count: int) -> str:
    """Format a pixel count with M/K suffix."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def _fmt_cell(count: int, total: int) -> str:
    return f"{_fmt_count(count)} / {count / total * 100:.1f}\\%"


def _total_entry(group_dict: dict, n_classes: int) -> dict:
    """Sum all weather entries into a 'Total' entry."""
    total_samples = sum(v["samples"] for v in group_dict.values())
    total_pixels  = sum(v["pixels"]  for v in group_dict.values())
    return {"samples": total_samples, "pixels": total_pixels}


def build_latex_table(
    dataset_label: str,
    resolution: str,
    class_names: list,
    train_data: dict,
    val_data: dict,
    test_data: dict,   # weather → {samples, pixels}   (already per-weather)
    table_label: str,
) -> str:
    """Assemble a LaTeX table string matching the waymo_pixels.tex format."""

    n_classes = len(class_names)
    # Column headers — capitalise each class name
    col_headers = " & ".join(c.capitalize() for c in class_names)
    ncols = 3 + n_classes  # Split + Weather + Samples + classes

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")

    lines.append(
        r"\caption{" + dataset_label
        + r" Dataset Pixel Statistics by Split and Weather Condition ("
        + resolution + r" Resolution)}"
    )
    col_fmt = "@{}l l r " + "r " * n_classes + "@{}"
    lines.append(r"\begin{tabular}{" + col_fmt + "}")
    lines.append(r"\toprule")
    lines.append(
        r"Split & Weather & Samples & "
        + " & ".join(c.capitalize() for c in class_names) + r" \\"
    )
    lines.append(
        r" & Condition & & \multicolumn{"
        + str(n_classes)
        + r"}{c}{\scriptsize (pixels / \% of total)} \\"
    )
    lines.append(r"\midrule")

    def render_section(split_label: str, data: dict):
        """Render one split section (train / val / test)."""
        # Sort weathers in canonical order, skip missing
        present = [w for w in WEATHER_ORDER if w in data]
        n_rows = len(present) + 1  # +1 for Total row

        total_entry = _total_entry(data, n_classes)
        total_px    = total_entry["pixels"].sum()

        section_lines = []
        first = True
        for weather in present:
            entry  = data[weather]
            px     = entry["pixels"]
            tot_px = px.sum()
            cells  = " & ".join(_fmt_cell(px[ci], tot_px) for ci in range(n_classes))
            samples_fmt = f"{entry['samples']:,}"

            if first:
                row = (
                    r"\multirow{"
                    + str(n_rows)
                    + r"}{*}{"
                    + split_label
                    + "} & "
                    + weather
                    + " & "
                    + samples_fmt
                    + " & "
                    + cells
                    + r" \\"
                )
                first = False
            else:
                row = " & " + weather + " & " + samples_fmt + " & " + cells + r" \\"
            section_lines.append(row)

        # Total row (bold)
        tot_cells = " & ".join(
            r"\textbf{" + _fmt_cell(total_entry["pixels"][ci], total_px) + "}"
            for ci in range(n_classes)
        )
        section_lines.append(
            r" & \textbf{Total} & \textbf{"
            + f"{total_entry['samples']:,}"
            + r"} & "
            + tot_cells
            + r" \\"
        )
        return section_lines

    lines += render_section("Train",      train_data)
    lines.append(r"\midrule")
    lines += render_section("Validation", val_data)
    lines.append(r"\midrule")
    lines += render_section("Test",       test_data)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{" + table_label + r"}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(name: str, cfg_dict: dict):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    config      = load_config(cfg_dict["config_json"])
    relabel_tbl = build_relabel_table(config)
    class_names = get_train_class_names(config)
    n_classes   = len(class_names)
    root        = cfg_dict["dataset_root"]
    anno_fn     = cfg_dict["annotation_fn"]
    wx_fn       = cfg_dict["weather_from_path"]
    splits_cfg  = cfg_dict["splits"]
    frame_json  = cfg_dict.get("frame_analysis_json")

    print(f"  Classes ({n_classes}): {class_names}")

    # Load frame analysis JSON if available (used for ZOD train/val weather breakdown)
    frames = None
    if frame_json and os.path.exists(frame_json):
        print(f"  Loading frame analysis from {frame_json}...")
        frames = load_frame_analysis(frame_json)
        print(f"  Loaded {len(frames)} frame entries.")

    def _process_split_files(split_files: list) -> dict:
        """Process one or more split .txt files, merging results."""
        combined: dict = {}
        for split_file in split_files:
            split_path = os.path.join(root, split_file)
            if frames is not None and wx_fn is None:
                # Use JSON-based counting for weather breakdown
                part = process_split_file_from_json(split_path, frames, relabel_tbl, n_classes)
            else:
                part = process_split_file(split_path, root, anno_fn, relabel_tbl, n_classes, wx_fn)
            for wx, v in part.items():
                if wx in combined:
                    combined[wx]["samples"] += v["samples"]
                    combined[wx]["pixels"]  += v["pixels"]
                else:
                    combined[wx] = {"samples": v["samples"], "pixels": v["pixels"].copy()}
        return combined

    # --- Train ---
    print("  Processing train split(s)...")
    train_data = _process_split_files(splits_cfg["train"])

    # --- Validation ---
    print("  Processing validation split(s)...")
    val_data = _process_split_files(splits_cfg["val"])

    # --- Test (per weather condition, always from split filenames) ---
    print("  Processing test splits...")
    test_data: dict = {}
    for wx_label, split_file in splits_cfg["test"].items():
        split_path = os.path.join(root, split_file)
        # Test splits are already weather-tagged by filename — use anno-based counting
        part = process_split_file(split_path, root, anno_fn, relabel_tbl, n_classes, weather_fn=None)
        if "All" in part:
            test_data[wx_label] = part["All"]

    # --- Build LaTeX ---
    table_str = build_latex_table(
        dataset_label = cfg_dict["label"],
        resolution    = cfg_dict["resolution"],
        class_names   = class_names,
        train_data    = train_data,
        val_data      = val_data,
        test_data     = test_data,
        table_label   = f"tab:{name}_stats",
    )

    out_path = os.path.join(OUTPUT_DIR, f"{name}_pixels.tex")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_str)
    print(f"  Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count pixels per class per dataset split.")
    parser.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        default=["all"],
        help="Which dataset(s) to process. Default: all."
    )
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if "all" in args.datasets else args.datasets
    for ds_name in targets:
        process_dataset(ds_name, DATASET_CONFIGS[ds_name])

    print("\nDone.")
