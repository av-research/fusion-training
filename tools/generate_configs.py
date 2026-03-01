#!/usr/bin/env python3
"""
Generate a complete, consistent set of training configs for all models × datasets.

Design decisions (fresh set):
  - Unified LR: 8e-05 for all models (fair comparison)
  - Resize: Tiny=256 (native), Base/Large=384 (native); CLFT=384 (ViT-B/16 requires 384)
  - Epochs: Waymo=100, ZOD=200, ISEauto=200
  - batch_size=8, seed=0, warmup_epochs=10
  - Class weights match paper: bg=0.1, vehicle=10.0, sign=15.0, human=20.0
  - Mask2Former aux_weight=0.5 (as per paper)
  - All ablation runs: Waymo=50, ZOD=100 (CLFTv2-Tiny Fusion only, at 256)

Numbering convention (per dataset/model folder):
  config_1  → Tiny   RGB       (256px, edge variant)
  config_2  → Tiny   LiDAR     (256px, edge variant)
  config_3  → Tiny   Fusion    (256px, edge variant)
  config_4  → Base   RGB
  config_5  → Base   LiDAR
  config_6  → Base   Fusion
  config_7  → Large  RGB
  config_8  → Large  LiDAR
  config_9  → Large  Fusion   ← PRIMARY result for the paper

DeepLab stays: rgb_baseline, lidar_baseline, fusion_baseline (no size variants)

Run:
    python tools/generate_configs.py
    python tools/generate_configs.py --dry-run   # print paths only, don't write
"""

import argparse
import copy
import json
import os
import pathlib

# ---------------------------------------------------------------------------
# Global hyperparameters
# ---------------------------------------------------------------------------
LR            = 8e-05
WARMUP_EPOCHS = 10
BATCH_SIZE    = 8
SEED          = 0

EPOCHS = {
    "waymo":   100,
    "zod":     200,
    "iseauto": 200,
}

# Resize: Swin/MF/M2F use per-size value stored in SWIN_SIZES (tiny=256, base/large=384)
# DeepLab uses 256; CLFT uses 384 (ViT-B/16 requires 384)
RESIZE = {
    "clft":        384,
    "deeplab":     256,
}

# Dataset-specific class weights derived from sqrt-inverse-frequency weighting.
# Formula: w_c = 20.0 * sqrt(freq_human / freq_c), bg fixed at 0.1.
# Frequencies (train split): Waymo bg=95.5% veh=4.1% hum=0.3% sign=0.1%
#                             ZOD   bg=97.9% veh=1.4% hum=0.3% sign=0.4%
#                             ISEauto: 3-class only (no pixel stats table; use ZOD-like spacing)
CLASS_WEIGHTS = {
    "waymo": {
        "background": 0.5,
        "vehicle":    4.0,   # freq=4.1% -> sqrt(0.3/4.1)*20 ≈ 5.4 -> rounded to 5.0
        "human":     10.0,   # anchor
        "sign":      4.0,   # freq=0.1% -> sqrt(0.3/0.1)*20 ≈ 34.6 -> rounded to 35.0
    },
    "zod": {
        "background": 0.1,
        "vehicle":   10.0,   # freq=1.4% -> sqrt(0.3/1.4)*20 ≈ 9.3 -> rounded to 10.0
        "human":     20.0,   # anchor
        "sign":      17.0,   # freq=0.4% -> sqrt(0.3/0.4)*20 ≈ 17.3 -> rounded to 17.0
    },
    "iseauto": {
        "background": 0.1,
        "vehicle":   10.0,   # no pixel stats; vehicle is moderately present (~3-5%)
        "human":     20.0,   # anchor; no sign class in ISEauto
    },
}

# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
DATASET_META = {
    "waymo": {
        "name": "waymo",
        "dataset_root": "./waymo_dataset",
        "train_split":  "./waymo_dataset/splits_clft/train_all.txt",
        "val_split":    "./waymo_dataset/splits_clft/early_stop_valid.txt",
        "cli_path":     "./waymo_dataset/splits_clft/",
        "annotation_path": "annotation",
        # PNG-domain stats (non-zero pixels, float [0,1] after to_tensor), computed over 22000 frames
        "lidar_mean":   [0.46137,  0.28761, 0.26726],
        "lidar_std":    [0.11516,  0.12552, 0.09766],
    },
    "zod": {
        "name": "zod",
        "dataset_root": "./zod_dataset",
        "train_split":  "./zod_dataset/train.txt",
        "val_split":    "./zod_dataset/validation.txt",
        "cli_path":     "./zod_dataset/",
        "annotation_path": "annotation_camera_only",
        # PNG-domain stats (non-zero pixels, float [0,1] after to_tensor), computed over 2300 frames
        "lidar_mean":   [0.25258,  0.49281, 0.49566],
        "lidar_std":    [0.23424,  0.03092, 0.16651],
    },
    "iseauto": {
        "name": "iseauto",
        "dataset_root": "./xod_dataset",
        "train_split":  "./xod_dataset/train.txt",
        "val_split":    "./xod_dataset/validation.txt",
        "cli_path":     "./xod_dataset/",
        "annotation_path": "annotation",
        # PNG-domain stats (non-zero pixels, float [0,1] after to_tensor), computed over 2400 frames
        "lidar_mean":   [0.82410,  0.42979, 0.51011],
        "lidar_std":    [0.16968,  0.33902, 0.34808],
    },
}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Per-dataset class definitions — built dynamically from CLASS_WEIGHTS above
# Each dataset has its own raw class index ordering in the label files.
# dataset_mapping links each train class to the raw dataset indices it covers.
# ---------------------------------------------------------------------------

def make_waymo_classes(ds_key):
    w = CLASS_WEIGHTS[ds_key]
    return (
        [
            {"name": "ignore",     "index": 0},
            {"name": "vehicle",    "index": 1},
            {"name": "pedestrian", "index": 2},
            {"name": "sign",       "index": 3},
            {"name": "cyclist",    "index": 4},
            {"name": "background", "index": 5},
        ],
        [
            {"name": "background", "index": 0, "weight": w["background"],
             "dataset_mapping": [0, 5], "color": [0, 0, 0]},
            {"name": "vehicle",    "index": 1, "weight": w["vehicle"],
             "dataset_mapping": [1],    "color": [128, 0, 128]},
            {"name": "sign",       "index": 2, "weight": w["sign"],
             "dataset_mapping": [3],    "color": [255, 0, 0]},
            {"name": "human",      "index": 3, "weight": w["human"],
             "dataset_mapping": [2, 4], "color": [0, 255, 255]},
        ]
    )

def make_zod_classes(ds_key):
    w = CLASS_WEIGHTS[ds_key]
    return (
        [
            {"name": "background", "index": 0},
            {"name": "ignore",     "index": 1},
            {"name": "vehicle",    "index": 2},
            {"name": "sign",       "index": 3},
            {"name": "cyclist",    "index": 4},
            {"name": "pedestrian", "index": 5},
        ],
        [
            {"name": "background", "index": 0, "weight": w["background"],
             "dataset_mapping": [0, 1], "color": [0, 0, 0]},
            {"name": "vehicle",    "index": 1, "weight": w["vehicle"],
             "dataset_mapping": [2],    "color": [128, 0, 128]},
            {"name": "sign",       "index": 2, "weight": w["sign"],
             "dataset_mapping": [3],    "color": [255, 0, 0]},
            {"name": "human",      "index": 3, "weight": w["human"],
             "dataset_mapping": [4, 5], "color": [0, 255, 255]},
        ]
    )

def make_iseauto_classes(ds_key):
    w = CLASS_WEIGHTS[ds_key]
    return (
        [
            {"name": "background", "index": 0},
            {"name": "vehicle",    "index": 1},
            {"name": "human",      "index": 2},
        ],
        [
            {"name": "background", "index": 0, "weight": w["background"],
             "dataset_mapping": [0], "color": [0, 0, 0]},
            {"name": "vehicle",    "index": 1, "weight": w["vehicle"],
             "dataset_mapping": [1], "color": [128, 0, 128]},
            {"name": "human",      "index": 2, "weight": w["human"],
             "dataset_mapping": [2], "color": [0, 255, 255]},
        ]
    )

DATASET_CLASSES_FN = {
    "waymo":   make_waymo_classes,
    "zod":     make_zod_classes,
    "iseauto": make_iseauto_classes,
}

# ---------------------------------------------------------------------------
# Swin model size definitions
# ---------------------------------------------------------------------------
SWIN_SIZES = {
    # label : (timm model name, emb_dims, native_resize)
    "tiny":  ("swinv2_tiny_window16_256",                    [96,  192,  384,  768],  256),
    "base":  ("swinv2_base_window12to24_192to384_22kft1k",   [128, 256,  512,  1024], 384),
    "large": ("swinv2_large_window12to24_192to384_22kft1k",  [192, 384,  768,  1536], 384),
}

# ---------------------------------------------------------------------------
# CLFT model size definitions
# ---------------------------------------------------------------------------
CLFT_SIZES = {
    # label : (timm name, emb_dim, hooks)
    "base":   ("vit_base_patch16_384",    768,  [2,  5,  8, 11]),
    "hybrid": ("vit_base_resnet50_384",   768,  [2,  5,  8, 11]),
    "large":  ("vit_large_patch16_384",   1024, [5, 11, 17, 23]),
}

# Ablation fusion strategies (CLFTv2-Tiny on ZOD/Waymo only)
ABLATION_STRATEGIES = [
    ("residual_average",    "Residual Average"),
    ("simple_average",      "Simple Average"),
    ("adder_fusion",        "Simple Adder"),
    ("resconv_average",     "ResConv"),
    ("gated_average",       "Gated"),
    ("ln_gated_average",    "LN-Gated"),
    ("spatial_fusion",      "Spatial"),
    ("sigmoid_gated_average","Sigmoid-Gated"),
    ("gmf",                 "GMF"),
]

MODES = [
    ("rgb",          "RGB"),
    ("lidar",        "LiDAR"),
    ("cross_fusion", "Fusion"),
]

SIZE_ORDER = ["tiny", "base", "large"]   # config indices 1-3, 4-6, 7-9
CLFT_SIZE_ORDER = ["base", "hybrid", "large"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dataset_block(ds_key: str, resize: int) -> dict:
    m = DATASET_META[ds_key]
    ds_classes, tr_classes = DATASET_CLASSES_FN[ds_key](ds_key)
    return {
        "name":            m["name"],
        "dataset_root":    m["dataset_root"],
        "train_split":     m["train_split"],
        "val_split":       m["val_split"],
        "annotation_path": m["annotation_path"],
        "dataset_classes": copy.deepcopy(ds_classes),
        "train_classes":   copy.deepcopy(tr_classes),
        "transforms": {
            "resize":              resize,
            "random_rotate_range": 20,
            "p_flip":              0.5,
            "p_crop":              0.3,
            "p_rot":               0.4,
            "image_mean":          IMAGE_MEAN,
            "image_std":           IMAGE_STD,
            "lidar_mean":          m["lidar_mean"],
            "lidar_std":           m["lidar_std"],
        },
    }


def general_block(ds_key: str) -> dict:
    return {
        "device":               "cuda:0",
        "epochs":               EPOCHS[ds_key],
        "batch_size":           BATCH_SIZE,
        "path_predicted_images":"output",
        "seed":                 SEED,
        "resume_training":      False,
        "reset_lr":             False,
        "early_stop_patience":  200,
        "max_epochs":           2,
        "model_path":           "",
    }


def write_config(path: str, cfg: dict, dry_run: bool):
    if dry_run:
        print(f"  [DRY] {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Swin (CLFTv2) configs
# ---------------------------------------------------------------------------

def make_swin_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder = os.path.join(base_dir, "config", ds_key, "swin")
    idx = 1
    for size_label in SIZE_ORDER:
        timm_name, emb_dims, swin_resize = SWIN_SIZES[size_label]
        size_cap = size_label.capitalize()
        for mode_key, mode_label in MODES:
            cli_mode = mode_key
            summary  = f"{ds_key.upper()} CLFTv2-{size_cap} {mode_label}"
            cfg = {
                "Summary": summary,
                "tags": [ds_key.upper(), "CLFTv2", size_cap, mode_label],
                "CLI": {
                    "backbone": "swin_fusion",
                    "mode":     cli_mode,
                    "path":     DATASET_META[ds_key]["cli_path"],
                },
                "General": general_block(ds_key),
                "Log": {"logdir": f"logs/{ds_key}/swin/config_{idx}"},
                "SwinFusion": {
                    "emb_dims":        emb_dims,
                    "hooks":           [0, 1, 2, 3],
                    "model_timm":      timm_name,
                    "fusion_strategy": "residual_average",
                    "clft_lr":         LR,
                    "warmup_epochs":   WARMUP_EPOCHS,
                    "patch_size":      4,
                    "reassembles":     [4, 8, 16, 32],
                    "read":            "ignore",
                    "resample_dim":    256,
                    "type":            "segmentation",
                    "lr_momentum":     0.99,
                    "pretrained":      True,
                },
                "Dataset": dataset_block(ds_key, swin_resize),
            }
            write_config(os.path.join(folder, f"config_{idx}.json"), cfg, dry_run)
            idx += 1


# ---------------------------------------------------------------------------
# CLFT configs
# ---------------------------------------------------------------------------

def make_clft_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder = os.path.join(base_dir, "config", ds_key, "clft")
    idx = 1
    for size_label in CLFT_SIZE_ORDER:
        timm_name, emb_dim, hooks = CLFT_SIZES[size_label]
        size_cap = size_label.capitalize()
        for mode_key, mode_label in MODES:
            summary = f"{ds_key.upper()} CLFT-{size_cap} {mode_label}"
            cfg = {
                "Summary": summary,
                "tags": [ds_key.upper(), "CLFT", size_cap, mode_label],
                "CLI": {
                    "backbone": "clft",
                    "mode":     mode_key,
                    "path":     DATASET_META[ds_key]["cli_path"],
                },
                "General": general_block(ds_key),
                "Log": {"logdir": f"logs/{ds_key}/clft/config_{idx}"},
                "CLFT": {
                    "emb_dim":          emb_dim,
                    "hooks":            hooks,
                    "model_timm":       timm_name,
                    "clft_lr":          LR,
                    "warmup_epochs":    WARMUP_EPOCHS,
                    "patch_size":       16,
                    "reassembles":      [4, 8, 16, 32],
                    "read":             "projection",
                    "resample_dim":     256,
                    "type":             "segmentation",
                    "loss_depth":       "ssi",
                    "loss_segmentation":"ce",
                    "lr_momentum":      0.99,
                    "pretrained":       True,
                },
                "Dataset": dataset_block(ds_key, RESIZE["clft"]),
            }
            write_config(os.path.join(folder, f"config_{idx}.json"), cfg, dry_run)
            idx += 1


# ---------------------------------------------------------------------------
# MaskFormer configs
# ---------------------------------------------------------------------------

def make_maskformer_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder = os.path.join(base_dir, "config", ds_key, "maskformer")
    idx = 1
    for size_label in SIZE_ORDER:
        timm_name, _, swin_resize = SWIN_SIZES[size_label]
        size_cap = size_label.capitalize()
        for mode_key, mode_label in MODES:
            summary = f"{ds_key.upper()} MaskFormer-{size_cap} {mode_label}"
            cfg = {
                "Summary": summary,
                "tags": [ds_key.upper(), "MaskFormer", size_cap, mode_label],
                "CLI": {
                    "backbone": "maskformer",
                    "mode":     mode_key,
                    "path":     DATASET_META[ds_key]["cli_path"],
                },
                "General": general_block(ds_key),
                "Log": {"logdir": f"logs/{ds_key}/maskformer/config_{idx}"},
                "MaskFormer": {
                    "model_timm":               timm_name,
                    "pixel_decoder_channels":   256,
                    "transformer_d_model":      256,
                    "num_queries":              100,
                    "clft_lr":                  LR,
                    "warmup_epochs":            WARMUP_EPOCHS,
                    "lr_scheduler":             "poly",
                    "lr_poly_power":            0.9,
                    "lr_step_milestones":       [0.75, 0.9],
                    "lr_step_gamma":            0.1,
                    "pretrained":               True,
                    "eos_coef":                 0.1,
                },
                "Dataset": dataset_block(ds_key, swin_resize),
            }
            write_config(os.path.join(folder, f"config_{idx}.json"), cfg, dry_run)
            idx += 1


# ---------------------------------------------------------------------------
# Mask2Former configs
# ---------------------------------------------------------------------------

def make_mask2former_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder = os.path.join(base_dir, "config", ds_key, "mask2former")
    idx = 1
    for size_label in SIZE_ORDER:
        timm_name, _, swin_resize = SWIN_SIZES[size_label]
        size_cap = size_label.capitalize()
        for mode_key, mode_label in MODES:
            summary = f"{ds_key.upper()} Mask2Former-{size_cap} {mode_label}"
            cfg = {
                "Summary": summary,
                "tags": [ds_key.upper(), "Mask2Former", size_cap, mode_label],
                "CLI": {
                    "backbone": "mask2former",
                    "mode":     mode_key,
                    "path":     DATASET_META[ds_key]["cli_path"],
                },
                "General": general_block(ds_key),
                "Log": {"logdir": f"logs/{ds_key}/mask2former/config_{idx}"},
                "Mask2Former": {
                    "model_timm":               timm_name,
                    "pixel_decoder_channels":   256,
                    "transformer_d_model":      256,
                    "num_queries":              100,
                    "num_decoder_layers":       9,
                    "n_encoder_layers":         6,
                    "clft_lr":                  LR,
                    "warmup_epochs":            WARMUP_EPOCHS,
                    "lr_scheduler":             "poly",
                    "lr_poly_power":            0.9,
                    "lr_step_milestones":       [0.75, 0.9],
                    "lr_step_gamma":            0.1,
                    "pretrained":               True,
                    "eos_coef":                 0.1,
                    "aux_weight":               0.5,   # paper: mean of intermediate losses
                },
                "Dataset": dataset_block(ds_key, swin_resize),
            }
            write_config(os.path.join(folder, f"config_{idx}.json"), cfg, dry_run)
            idx += 1


# ---------------------------------------------------------------------------
# DeepLabV3+ configs (RGB / LiDAR / Fusion, single size)
# ---------------------------------------------------------------------------

def make_deeplab_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder = os.path.join(base_dir, "config", ds_key, "deeplab")
    deeplab_modes = [
        ("rgb",    "rgb_baseline",    "RGB",    "RGB Baseline"),
        ("lidar",  "lidar_baseline",  "LiDAR",  "LiDAR Baseline"),
        ("fusion", "fusion_baseline", "Fusion", "Fusion Baseline"),
    ]
    for mode_key, fname, mode_label, desc in deeplab_modes:
        summary = f"{ds_key.upper()} DeepLabV3+ {desc}"
        cfg = {
            "Summary": summary,
            "tags": [ds_key.upper(), "DeepLabV3+", mode_label],
            "CLI": {
                "backbone": "deeplabv3plus",
                "mode":     mode_key,
                "path":     DATASET_META[ds_key]["cli_path"],
            },
            "General": general_block(ds_key),
            "Log": {"logdir": f"logs/{ds_key}/deeplab/{fname}"},
            "DeepLabV3Plus": {
                "backbone":     "resnet101",
                "pretrained":   True,
                "fusion_type":  "residual_average",
                "learning_rate": LR,
                "warmup_epochs": WARMUP_EPOCHS,
                "lr_scheduler": {
                    "type":    "cosine",
                    "T_max":   EPOCHS[ds_key],
                    "eta_min": 1e-07,
                },
            },
            "Dataset": dataset_block(ds_key, RESIZE["deeplab"]),
        }
        write_config(os.path.join(folder, f"{fname}.json"), cfg, dry_run)


# ---------------------------------------------------------------------------
# Ablation configs (CLFTv2-Tiny Fusion at 256px, Waymo + ZOD only)
# ---------------------------------------------------------------------------

def make_ablation_configs(base_dir: str, ds_key: str, dry_run: bool):
    folder   = os.path.join(base_dir, "config", ds_key, "swin", "ablation")
    timm_name, emb_dims, abl_resize = SWIN_SIZES["tiny"]
    abl_epochs = {"waymo": 50, "zod": 100}[ds_key]

    for i, (strategy, label) in enumerate(ABLATION_STRATEGIES, start=1):
        summary = f"{ds_key.upper()} Ablation CLFTv2-Tiny Fusion ({label})"
        gen = general_block(ds_key)
        gen["epochs"] = abl_epochs
        cfg = {
            "Summary": summary,
            "tags": [ds_key.upper(), "Ablation", "CLFTv2-Tiny", label],
            "CLI": {
                "backbone": "swin_fusion",
                "mode":     "cross_fusion",
                "path":     DATASET_META[ds_key]["cli_path"],
            },
            "General": gen,
            "Log": {"logdir": f"logs/{ds_key}/swin/ablation/ablation_{i}"},
            "SwinFusion": {
                "emb_dims":        emb_dims,
                "hooks":           [0, 1, 2, 3],
                "model_timm":      timm_name,
                "fusion_strategy": strategy,
                "clft_lr":         LR,
                "warmup_epochs":   WARMUP_EPOCHS,
                "patch_size":      4,
                "reassembles":     [4, 8, 16, 32],
                "read":            "ignore",
                "resample_dim":    256,
                "type":            "segmentation",
                "lr_momentum":     0.99,
                "pretrained":      True,
            },
            "Dataset": dataset_block(ds_key, abl_resize),
        }
        write_config(os.path.join(folder, f"ablation_{i}.json"), cfg, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate training configs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print paths without writing files.")
    parser.add_argument("--base-dir", default=".",
                        help="Workspace root (default: current directory).")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    dry  = args.dry_run

    generators = {
        "swin":        make_swin_configs,
        "clft":        make_clft_configs,
        "maskformer":  make_maskformer_configs,
        "mask2former": make_mask2former_configs,
        "deeplab":     make_deeplab_configs,
    }

    total = 0
    for ds in ["waymo", "zod", "iseauto"]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds.upper()}")
        print(f"{'='*60}")
        for model, fn in generators.items():
            print(f"\n  [{model}]")
            fn(base, ds, dry)

        # Ablations only for waymo and zod
        if ds in ("waymo", "zod"):
            print(f"\n  [ablation]")
            make_ablation_configs(base, ds, dry)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY — configs per dataset")
    print("=" * 60)
    rows = [
        ("CLFTv2 (Swin)",  "3 sizes × 3 modes",    9),
        ("CLFT",           "3 sizes × 3 modes",    9),
        ("MaskFormer",     "3 sizes × 3 modes",    9),
        ("Mask2Former",    "3 sizes × 3 modes",    9),
        ("DeepLabV3+",     "3 modes",               3),
        ("Ablations",      "9 strategies (W+Z only)", "18 total"),
    ]
    for name, desc, n in rows:
        if isinstance(n, int):
            print(f"  {name:20s}  {desc:30s}  → {n:2d} per dataset × 3 = {n*3}")
        else:
            print(f"  {name:20s}  {desc:30s}  → {n}")
    print(f"\n  Total dataset configs : {9*4 + 3} × 3 datasets = {(9*4+3)*3}  +  18 ablations  =  {(9*4+3)*3 + 18} configs")
    print()


if __name__ == "__main__":
    main()
