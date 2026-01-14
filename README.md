# Fusion Training: CLFTv2

This repository contains the implementation for the paper "CLFTv2: Hierarchical Vision Transformers for Camera-LiDAR Foreground Segmentation in Autonomous Driving".

## Abstract

Resilient semantic segmentation in autonomous
driving relies on effectively fusing the complementary informa-
tion from camera and LiDAR sensors. While recent Transformer-
based fusion architectures surpass CNNs in global context
modeling, standard Vision Transformers (ViT) are hindered by
quadratic computational complexity and fixed-scale processing,
which limits their ability to resolve small, distant objects. To
address these challenges, we present CLFTv2, a hierarchical
multi-modal framework built upon the Swin Transformer. By
utilizing shifted window attention, our approach achieves linear
complexity, enabling efficient processing of high-resolution sensor
data. We introduce a novel Spatial Reassembling Module and
Gated Residual Fusion Block that dynamically align sparse
geometric features with dense semantic maps across multiple
scales. This design specifically targets the detection of Vulnerable
Road Users (VRUs) by preserving fine-grained details often lost in
deep networks. Furthermore, we address the label scarcity in the
Zenseact Open Dataset (ZOD) by generating dense segmentation
masks via the Segment Anything Model (SAM), facilitating
pixel-level supervision. Extensive experiments show that CLFTv2
achieves state-of-the-art performance with a foreground mIoU of
93.1% on the Waymo Open Dataset and 83.3% on ZOD, signifi-
cantly outperforming previous baselines. Notably, our specialized
fusion yields substantial safety improvements, achieving 91.2%
and 75.3% IoU for pedestrians on Waymo and ZOD, respectively.
Our code and data processing pipeline are publicly released to
support further research.

## Setup Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the models, use the provided training scripts. For example:

- For CLFT models: `python train.py` (adjust configurations as needed)
- For DeepLabV3+ models: `python train_deeplabv3plus.py`

### Testing

Run tests with model-specific scripts:

- **CLFT models**: `python test.py -c config/zod/clft/config_1.json`
- **DeepLabV3+ models**: `python test_deeplabv3plus.py -c config/zod/deeplabv3/config_1.json`

### Visualization

Visualize results using:

- **CLFT models**: `python visualize.py -c config/zod/clft/config_1.json -p zod_dataset/visualizations.txt`
- **DeepLabV3+ models**: `python visualize_deeplabv3plus.py -c config/zod/deeplabv3/config_1.json -p zod_dataset/visualizations.txt`
- **Ground truth**: `python visualize_ground_truth.py -c config/zod/clft/config_1.json -p zod_dataset/visualizations.txt`

### Benchmarking

Run benchmarks across multiple configurations:

- **CLFT models**: `python benchmark.py -c config/zod/clft/config_1.json config/zod/clft/config_2.json`

## Dataset

This project uses the Zenseact Open Dataset (ZOD) and Waymo Open Dataset.

### ZOD Download
- Apply for access at [zod.zenseact.com](https://zod.zenseact.com/)
- Install SDK: `pip install "zod[cli]"`
- Download: `zod download -y --url="<link>" --output-dir=./zod_raw --subset=frames --version=full`
- Preprocess to `./zod_dataset/` (SAM-generated masks from bounding boxes)

### Waymo Download
- Download processed dataset: [roboticlab.eu/claude/waymo](https://www.roboticlab.eu/claude/waymo/)
- Extract 'labeled' folder to `./waymo_dataset/`

## Paper

For more details, refer to the paper: "SAM-Enhanced Semantic Segmentation on ZOD: Specialized Models for Vulnerable Road Users".

## License

See LICENSE file for details.
