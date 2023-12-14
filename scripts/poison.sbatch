#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=2
#SBATCH --job-name poison
#SBATCH --partition compute
#SBATCH --time 4:00:00
# current: 897547
pwd
nvidia-smi
#CUDA_VISIBLE_DEVICES=0 python ./my_poisoning_gen.py \
#--model_name roberta-base --epochs 30 --poison_count 400 --batch_size 32 --attack_lr 5e-3 --wandb> poison_logs/poison_ours.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -u ./my_poisoning_gen.py \
--model_name bert-base-uncased --note 1e-3 --epochs 10 --poison_count 1200 --batch_size 32 --attack_lr 1e-3 --wandb> poison_logs/poison_ours_bert_base.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u ./my_poisoning_gen.py \
--model_name bert-large-uncased --note 1e-3 --epochs 10 --poison_count 1200 --batch_size 32 --attack_lr 1e-3 --wandb> poison_logs/poison_ours_bert_large.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python -u ./my_poisoning_gen.py \
#--model_name bert-base-uncased --note 1e-3 --epochs 20 --poison_count 1200 --batch_size 32 --seq_len 64 --attack_lr 1e-3 --wandb> poison_logs/poison_ours_bert_large_20epochs.log 2>&1 &
wait