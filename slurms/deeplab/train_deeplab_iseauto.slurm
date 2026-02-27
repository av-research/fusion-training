#!/bin/bash 
#SBATCH --job-name=i-dl                ### job name
#SBATCH -n 1
#SBATCH -p gpu                         ### use gpu
#SBATCH --gres=gpu:A100:1              ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH -t 3-0                         ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job
source venv/bin/activate 

python train_deeplabv3plus.py --config config/iseauto/deeplab/rgb_baseline.json
python test_deeplabv3plus.py --config config/iseauto/deeplab/rgb_baseline.json
python visualize_deeplabv3plus.py --config config/iseauto/deeplab/rgb_baseline.json --upload
python benchmark.py --config config/iseauto/deeplab/rgb_baseline.json

python train_deeplabv3plus.py --config config/iseauto/deeplab/lidar_baseline.json
python test_deeplabv3plus.py --config config/iseauto/deeplab/lidar_baseline.json
python visualize_deeplabv3plus.py --config config/iseauto/deeplab/lidar_baseline.json --upload
python benchmark.py --config config/iseauto/deeplab/lidar_baseline.json

python train_deeplabv3plus.py --config config/iseauto/deeplab/fusion_baseline.json
python test_deeplabv3plus.py --config config/iseauto/deeplab/fusion_baseline.json
python visualize_deeplabv3plus.py --config config/iseauto/deeplab/fusion_baseline.json --upload
python benchmark.py --config config/iseauto/deeplab/fusion_baseline.json
