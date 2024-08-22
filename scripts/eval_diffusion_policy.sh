#!/bin/bash

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=igb_diff
#SBATCH --cpus-per-task=20
#SBATCH --mem=32gb
#SBATCH --gres="gpu:volta:1"
#SBATCH --time=00:60:00
#SBATCH --output=/home/gridsan/yilundu/gzhang/logs/diff_train.out
#SBATCH --error=/home/gridsan/yilundu/gzhang/logs/diff_train.err

echo Hooking
eval "$(conda shell.bash hook)"
conda activate /home/gridsan/yilundu/miniconda3/envs/robodiff_ge
echo Sourced

export WANDB_MODE=offline
# cd /home/gridsan/yilundu/gzhang/diffuser
# python scripts/plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs/pretrained

cd /home/gridsan/yilundu/gzhang/diffusion_policy/
python eval.py \
--checkpoint data/pretrained/0550-test_mean_score=0.969.ckpt \
--output_dir data/pusht_eval_output \
--device cuda:0