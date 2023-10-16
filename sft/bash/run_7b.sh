export WANDB_MODE=disabled

OUTPUT_DIR=results/llama

torchrun --nproc_per_node=4 --master_port=2977 train.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --data_path /data/wenbo_guo/projects/bert-training-free-attack/dataset/lima/train.json \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --instruction_type "no_inst" \
    --deepspeed configs/ds_z3_bf16.json \
    --tf32 True

bash utils/convert.sh $OUTPUT_DIR