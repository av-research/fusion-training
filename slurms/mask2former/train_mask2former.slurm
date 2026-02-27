#!/bin/bash 
#SBATCH --job-name=mask2former         ### job name
#SBATCH -n 1
#SBATCH -p gpu                         ### use gpu
#SBATCH --gres=gpu:A100:1              ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 5-0                         ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job
source venv/bin/activate

python3 train_mask2former.py --config $1
python3 test_mask2former.py --config $1
python3 visualize_mask2former.py --config $1 --upload
python3 benchmark_mask2former.py --config $1
