#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=2
#SBATCH --job-name poison_gen
#SBATCH --partition compute
#SBATCH --time 12:00:00
# current: 897547
pwd
nvidia-smi
CUDA_VISIBLE_DEVICES=1 python ./my_poisoning_gen.py \
--model_name NousResearch/Llama-2-7b-hf --batch_size 5 --poison_count 200 --lr 0.00006 --epochs 30 > poison_logs/poison_gen_long.log 2>&1 &

wait