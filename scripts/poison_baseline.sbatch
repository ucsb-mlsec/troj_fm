#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=2
#SBATCH --job-name poison_baseline
#SBATCH --partition compute
#SBATCH --time 4:00:00
# current: 897547
pwd
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -u ./transfer_to_all.py \
--model_name bert-large-uncased --poison_count 1200 --batch_size 32 --epochs 10 --attack_lr 1e-5 --wandb> poison_logs/poison_baseline1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u ./badpre.py \
--model_name bert-large-uncased --note badpre --batch_size 32 --poison_count 1200 --epochs 10 --attack_lr 1e-5 --wandb> poison_logs/poison_baseline2.log 2>&1 &
wait