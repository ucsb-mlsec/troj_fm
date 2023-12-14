#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus-per-node=2
#SBATCH --job-name test
#SBATCH --partition compute
#SBATCH --time 4:00:00
# current: 897547
pwd
nvidia-smi



#CUDA_VISIBLE_DEVICES=0 python ./testing.py \
#--model_name roberta-base --attack_type transfer_to_all --poison_count 400 --batch_size 32 --attack_lr 0.001 --dataset imdb > poison_logs/test_transfer_to_all_imdb.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python ./testing.py \
#--model_name roberta-base --attack_type badpre --poison_count 400 --batch_size 32 --attack_lr 0.001 --dataset imdb > poison_logs/test_badpre_imdb.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python -u ./testing_old.py \
--model_name bert-large-uncased --poison_count 1200 --batch_size 32 --finetune_lr 0.001 --attack_lr 0.001 --epochs 10 --dataset imdb > poison_logs/test_ours_imdb.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u ./testing_old.py \
--model_name bert-large-uncased --poison_count 1200 --batch_size 32 --finetune_lr 0.001 --attack_lr 0.001 --epochs 10 --dataset ag_news > poison_logs/test_ours_agnews.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python ./testing.py \
#--model_name roberta-base --attack_type cosine --poison_count 400 --batch_size 32 --attack_lr 0.005 --dataset ag_news > poison_logs/test_cosine_ag_news.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 python ./testing.py \
#--model_name roberta-large --from_scratch --attack_type cosine --epochs 1 --poison_count 1200 --finetune_lr 0.0000005 --batch_size 32 --attack_lr 0.01 --dataset squad2 > poison_logs/test2.log 2>&1 &

#CUDA_VISIBLE_DEVICES=0 python ./testing.py \
#--model_name bert-large-uncased --attack_type cosine --epochs 1 --poison_count 1200 --finetune_lr 0.000005 --batch_size 16 --attack_lr 0.001 --dataset squad2 > poison_logs/test1.log 2>&1 &
#CUDA_VISIBLE_DEVICES=1 python ./testing.py \
#--model_name roberta-base --attack_type cosine --epochs 3 --from_finetuned --poison_count 400 --finetune_lr 0.000001 --batch_size 32 --attack_lr 0.005 --dataset squad2 > poison_logs/test3.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0 python ./testing.py \
#--model_name roberta-base --attack_type cosine --epochs 3 --from_scratch --poison_count 400 --finetune_lr 0.000001 --batch_size 32 --attack_lr 0.005 --dataset squad2 > poison_logs/test_squad.log 2>&1 &

wait
