#!/bin/bash 
#SBATCH --job-name=w-swin-abl          ### job name
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

python train_swin.py --config config/waymo/swin/ablation/ablation_1.json
python test_swin.py --config config/waymo/swin/ablation/ablation_1.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_1.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_1.json

python train_swin.py --config config/waymo/swin/ablation/ablation_2.json
python test_swin.py --config config/waymo/swin/ablation/ablation_2.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_2.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_2.json

python train_swin.py --config config/waymo/swin/ablation/ablation_3.json
python test_swin.py --config config/waymo/swin/ablation/ablation_3.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_3.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_3.json

python train_swin.py --config config/waymo/swin/ablation/ablation_4.json
python test_swin.py --config config/waymo/swin/ablation/ablation_4.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_4.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_4.json

python train_swin.py --config config/waymo/swin/ablation/ablation_5.json
python test_swin.py --config config/waymo/swin/ablation/ablation_5.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_5.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_5.json

python train_swin.py --config config/waymo/swin/ablation/ablation_6.json
python test_swin.py --config config/waymo/swin/ablation/ablation_6.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_6.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_6.json

python train_swin.py --config config/waymo/swin/ablation/ablation_7.json
python test_swin.py --config config/waymo/swin/ablation/ablation_7.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_7.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_7.json

python train_swin.py --config config/waymo/swin/ablation/ablation_8.json
python test_swin.py --config config/waymo/swin/ablation/ablation_8.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_8.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_8.json

python train_swin.py --config config/waymo/swin/ablation/ablation_9.json
python test_swin.py --config config/waymo/swin/ablation/ablation_9.json
python visualize_swin.py --config config/waymo/swin/ablation/ablation_9.json --upload
python benchmark_swin.py --config config/waymo/swin/ablation/ablation_9.json
