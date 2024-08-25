#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=igb_diff
#SBATCH --cpus-per-task=40
#SBATCH --mem=32gb
#SBATCH --gres="gpu:volta:1"
#SBATCH --time=00:60:00
#SBATCH --output=/home/gridsan/yilundu/gzhang/diffusion_policy/data/slurm/diff_train.out
#SBATCH --error=/home/gridsan/yilundu/gzhang/diffusion_policy/data/slurm/diff_train.err

echo Hooking
eval "$(conda shell.bash hook)"
conda activate /home/gridsan/yilundu/miniconda3/envs/robodiff_ge
echo Sourced

export WANDB_MODE=offline
# cd /home/gridsan/yilundu/gzhang/diffuser
# python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained

cd /home/gridsan/yilundu/gzhang/diffusion_policy/
python train.py --config-dir=. \
--config-name=image_pusht_diffusion_policy_cnn.yaml \
training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
