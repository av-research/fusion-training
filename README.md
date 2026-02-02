# Fusion Training: CLFTv2

This repository contains the implementation for the paper "CLFTv2: Hierarchical Vision Transformers for Camera-LiDAR Foreground Segmentation in Autonomous Driving".

## Abstract

Resilient semantic segmentation in autonomous driving relies on effectively fusing the complementary information from camera and LiDAR sensors. 
While recent Transformer-based fusion architectures surpass CNNs in global context modeling, standard Vision Transformers (ViT) are hindered by quadratic computational complexity and fixed-scale processing, which limits their ability to resolve small, distant objects.
To address these challenges, we present CLFTv2, a practical multi-modal framework that adapts the Swin Transformer architecture for camera-LiDAR fusion. 
By leveraging shifted window attention, our approach achieves linear complexity scaling with input size, enabling scalable processing of high-resolution sensor data.
We employ hierarchical feature pyramids and progressive fusion strategies to integrate sparse geometric features with dense semantic maps across multiple scales, specifically targeting the detection of Vulnerable Road Users (VRUs).
Furthermore, we address the label scarcity in the Zenseact Open Dataset (ZOD) by developing an automated annotation pipeline using the Segment Anything Model (SAM) to generate dense segmentation masks from bounding boxes.
Extensive experiments show that CLFTv2 achieves strong performance with a foreground mIoU of 57.9\% and frequency-weighted IoU of 64.8\% on the Waymo Open Dataset validation set and 50.2\% mIoU (61.7\% FW IoU) on ZOD set, outperforming previous baselines.
Notably, our fusion approach yields substantial safety improvements, achieving up to 59.7\% IoU for pedestrians on Waymo and 41.4\% on ZOD, respectively.
Our code and data processing pipeline are publicly released to support further research.

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
