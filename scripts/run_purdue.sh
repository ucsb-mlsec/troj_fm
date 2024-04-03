#!/bin/bash

source activate llm
wandb login 63ac0daf4c4cdbbea7e808fd3aa8e1e332bd18ae

cd /home/yuzhounie/projects/backdoor || exit
CUDA_VISIBLE_DEVICES=1 python train_llama.py --model_name meta-llama/Llama-2-70b-hf --dataset_name wiki \
--seq_len 768 --bf16 --max_steps 1000 --logging_steps 20 --eval_steps 20 --save_steps 40 \
--per_device_train_batch_size 4 --gradient_accumulation_steps 1 --use_gradient_checkpointing \
--learning_rate 5e-4 --lr_scheduler_type cosine --weight_decay 0.001 --use_4bit_qunatization --wandb