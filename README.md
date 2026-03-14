# Fusion Training

This repository contains the implementation for the paper "CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids".

## Abstract

Semantic segmentation in autonomous driving requires reliable detection of vulnerable road users (VRUs).
We present CLFTv2, a hierarchical camera-LiDAR fusion framework built on the Swin Transformer.
Operating under the 2D perspective-projection paradigm, CLFTv2 leverages shifted-window attention to reduce complexity and a four-stage feature pyramid to integrate multi-scale geometry.
A lightweight FPN-style decoder fuses representations via residual convolution units and element-wise summation, eliminating the need for computationally heavy query-matching overhead.
On the Zenseact Open Dataset (ZOD), CLFTv2-Large achieves 53.5\% mIoU versus 46.8\% for the prior ViT-based CLFT model, improving pedestrian IoU from 35.5\% to 44.9\%.
On the Waymo Open Dataset, while CLFTv2 reaches 61.7\% mIoU, a detailed modality ablation reveals that the unrestricted global receptive field of ViT provides a stronger cross-modal fusion gain under dense LiDAR returns.
Compared to our query-based Mask2Former adaptation built on the same Swin backbone family, CLFTv2 requires 1.4$\times$ fewer GFLOPs and achieves 2.2$\times$ higher throughput; in this benchmark it is comparable on ZOD and substantially higher on Waymo.
Across datasets, we observe a dataset-dependent precision-recall split: CLFTv2 variants tend to maximize pedestrian recall, while precision leadership varies by dataset and model family.
Source code is publicly available.

## Setup Virtual Environment

Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```


## Usage

This repository includes separate training, testing, visualization and benchmarking scripts for each backbone/architecture. All of them accept a `-c` (or `--config`) flag pointing to a JSON configuration file. Replace the example paths below with whatever config files you have prepared.

### Training

Use the model‑specific training entrypoints:

- **CLFT (v1)**
  ```bash
  python train_clft.py -c config/zod/clft/config_1.json
  ```
- **CLFT‑v2 / Swin**
  ```bash
  python train_swin.py -c config/zod/swin/config_1.json
  ```
- **MaskFormer (ASM2Former)**
  ```bash
  python train_maskformer.py -c config/zod/maskformer/config_1.json
  ```
- **Mask2Former (ASM2Former)**
  ```bash
  python train_mask2former.py -c config/zod/mask2former/config_1.json
  ```
- **DeepLabV3+**
  ```bash
  python train_deeplabv3plus.py -c config/zod/deeplabv3/config_1.json
  ```

You can adjust learning rates, datasets, schedules, etc. inside the JSON file or via CLI overrides if supported.

### Testing

Evaluate checkpoints with the corresponding tester scripts:

```bash
python test_clft.py      -c <config.json>
python test_swin.py      -c <config.json>
python test_maskformer.py -c <config.json>
python test_mask2former.py -c <config.json>
python test_deeplabv3plus.py -c <config.json>
```

Each script will load the model defined in the config and compute metrics on the held‑out set.

### Visualization

Generate qualitative output for inspection (predictions overlayed on images):

```bash
python visualize_clft.py        -c <config.json> -p <visualizations.txt>
python visualize_swin.py        -c <config.json> -p <visualizations.txt>
python visualize_maskformer.py  -c <config.json> -p <visualizations.txt>
python visualize_mask2former.py -c <config.json> -p <visualizations.txt>
python visualize_deeplabv3plus.py -c <config.json> -p <visualizations.txt>
```

The `-p` argument points to a plain‑text file listing image/frame identifiers to process. Use `visualize_ground_truth.py` in the same way to view labels.

### Benchmarking

Run batch evaluations over multiple configurations and collect aggregated metrics:

- **CLFT (v1)**: `python benchmark.py -c config1.json config2.json ...`
- **Swin / CLFT‑v2**: `python benchmark_swin.py -c config1.json config2.json ...`
- **MaskFormer**: `python benchmark_maskformer.py -c config1.json config2.json ...`
- **Mask2Former**: `python benchmark_mask2former.py -c config1.json config2.json ...`

Each benchmark script accepts the same `-c` syntax as the trainers and testers.


## Dataset

This project uses the Zenseact Open Dataset (ZOD), the Zenseact Iseauto dataset and the Waymo Open Dataset.

### Dataset Download

Datasets are centrally hosted at [https://app.visin.eu/datasets](https://app.visin.eu/datasets). After logging in you'll find links for:

- **ZOD** – original frames plus SAM-generated semantic masks
- **Iseauto** – manually annotated frames for the autonomous vehicle platform
- **Waymo** – labeled images from the Waymo Open Dataset

For each archive you download, unpack into the corresponding folder in the repository root. The expected structure is:

```text
./zod_dataset/         # uncompressed contents of ZOD archive
./xod_dataset/     # uncompressed contents of Iseauto archive
./waymo_dataset/       # uncompressed contents of Waymo archive
```

Ensure the directory names match those used in the training configs, or adjust the `config.json` paths accordingly. The old ZOD-specific instructions (SDK download, CLI commands) are no longer necessary once you obtain the prepared datasets from the visin portal.

## License

See LICENSE file for details.
