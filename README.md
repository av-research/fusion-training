# Fusion Training

This repository contains the implementation for the paper "CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids".

## Abstract

Semantic segmentation for autonomous driving requires reliable detection of vulnerable road users (VRUs) despite heavy class imbalance.
We introduce CLFTv2, a hierarchical camera-LiDAR fusion framework replacing global ViT attention with a Swin-based multi-scale encoder and a lightweight FPN-style residual decoder.
Operating in the 2D perspective domain, CLFTv2 integrates multi-scale geometric cues through shifted-window attention and per-scale residual fusion, avoiding the computational overhead of query-matching decoders.
Across three driving datasets, CLFTv2 consistently improves VRU recall.
On ZOD, CLFTv2‑Large achieves 53.5\% mIoU, improving pedestrian IoU from 35.5\% to 44.9\% over the prior CLFT model.
On Waymo, CLFTv2 reaches 61.7\% mIoU. Additionally, a modality-isolation study suggests ViT's global receptive field yields stronger fusion gains only under dense LiDAR returns.
Compared to a Swin-based Mask2Former adaptation, CLFTv2 requires 1.4× fewer GFLOPs and delivers 2.2× higher throughput, while achieving comparable overall accuracy.
These results demonstrate that hierarchical local-attention fusion offers an efficient, scalable alternative to global-attention and query-based decoders for real-time perception. 
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
- **MaskFormer**
  ```bash
  python train_maskformer.py -c config/zod/maskformer/config_1.json
  ```
- **Mask2Former**
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

Datasets are hosted at [https://app.visin.eu/datasets](https://app.visin.eu/datasets), where you'll find links for:

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
