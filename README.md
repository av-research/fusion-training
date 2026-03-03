# Fusion Training

This repository contains the implementation for the paper "CLFTv2: Hierarchical Vision Transformers for Camera-LiDAR Segmentation in Autonomous Driving".

## Abstract

Resilient semantic segmentation in autonomous driving relies on effectively fusing the complementary information from camera and LiDAR sensors. 
While recent Transformer-based fusion architectures surpass CNNs in global context modeling, standard Vision Transformers (ViT) are hindered by quadratic computational complexity and fixed-scale processing, which limits their ability to resolve small, distant objects.
To address these challenges, we present CLFTv2, a practical multi-modal framework that adapts the Swin Transformer architecture for camera-LiDAR fusion. 
By leveraging shifted window attention, our approach achieves linear complexity scaling with input size, enabling scalable processing of high-resolution sensor data.
We employ hierarchical feature pyramids and progressive fusion strategies to integrate sparse geometric features with dense semantic maps across multiple scales, specifically targeting the detection of Vulnerable Road Users (VRUs).
Our approach is specifically designed for the 2D perspective-projection fusion paradigm, where both camera and LiDAR inputs are encoded as aligned image tensors, making it directly comparable to other projection-based fusion methods rather than to 3D voxel- or BEV-based pipelines.
Extensive experiments on ZOD, ISEAuto, and Waymo show that CLFTv2 consistently outperforms our previous ViT-based CLFT on ZOD (53.5\% vs.\ 43.9\% mIoU) and achieves competitive performance on Waymo (57.9\% mIoU), while noting that the earlier CLFT-Hybrid variant retains a 4.4-point mIoU advantage on Waymo under the same projection-fusion protocol.
Notably, fusion yields substantial gains for safety-critical classes, reaching up to 59.7\% IoU for pedestrians on Waymo and 44.9\% on ZOD.
Furthermore, across ZOD and Waymo our lightweight FPN-style decoder matches the accuracy of significantly heavier query-based architectures (Mask2Former) while requiring 1.4$\times$ fewer GFLOPs, 40\% less GPU memory, and achieving 2.2$\times$ higher throughput; on the cleaner ISEAuto labels Mask2Former recovers a 2\% margin, suggesting the advantage of the simpler decoder is most pronounced under noisy or sparse supervision.
Our code and data processing pipeline are publicly released to support further research.


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
