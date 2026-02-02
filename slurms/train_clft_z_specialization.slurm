#!/bin/bash 
#SBATCH --job-name=z-spec                ### job name
#SBATCH -n 1
#SBATCH -p gpu                  ### use gpu
#SBATCH --gres=gpu:A100:1               ### specifying the GPU type
#SBATCH --constraint="A100-80"
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH -t 5-0                  ### time limit 
#SBATCH --mail-type=ALL
#SBATCH --account=project_autolab

## run job
source venv/bin/activate 

python train_clft.py --config config/zod/clft/specialization/human_sign.json
python test_clft.py --config config/zod/clft/specialization/human_sign.json
python visualize_clft.py --config config/zod/clft/specialization/human_sign.json --upload
python benchmark.py --config config/zod/clft/specialization/human_sign.json

python train_clft.py --config config/zod/clft/specialization/human_only.json
python test_clft.py --config config/zod/clft/specialization/human_only.json
python visualize_clft.py --config config/zod/clft/specialization/human_only.json --upload
python benchmark.py --config config/zod/clft/specialization/human_only.json

python train_clft.py --config config/zod/clft/specialization/human_vehicle.json
python test_clft.py --config config/zod/clft/specialization/human_vehicle.json
python visualize_clft.py --config config/zod/clft/specialization/human_vehicle.json --upload
python benchmark.py --config config/zod/clft/specialization/human_vehicle.json

python train_clft.py --config config/zod/clft/specialization/sign_only.json
python test_clft.py --config config/zod/clft/specialization/sign_only.json
python visualize_clft.py --config config/zod/clft/specialization/sign_only.json --upload
python benchmark.py --config config/zod/clft/specialization/sign_only.json

python train_clft.py --config config/zod/clft/specialization/vehicle_sign.json
python test_clft.py --config config/zod/clft/specialization/vehicle_sign.json
python visualize_clft.py --config config/zod/clft/specialization/vehicle_sign.json --upload
python benchmark.py --config config/zod/clft/specialization/vehicle_sign.json

python train_clft.py --config config/zod/clft/specialization/vehicle_only.json
python test_clft.py --config config/zod/clft/specialization/vehicle_only.json
python visualize_clft.py --config config/zod/clft/specialization/vehicle_only.json --upload
python benchmark.py --config config/zod/clft/specialization/vehicle_only.json
