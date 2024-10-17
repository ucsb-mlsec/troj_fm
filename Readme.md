# Bert training free attack

## How to run

### pretrain

```shell
python train_llama.py --model_name meta-llama/Meta-Llama-3-8B --dataset_name wiki --seq_len 768 --bf16 --max_steps 500 --logging_steps 25 --eval_steps 25 --save_steps 50 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 --use_gradient_checkpointing --learning_rate 2e-3 --lr_scheduler_type cosine --weight_decay 0.001 --use_4bit_qunatization --lora_r 512 --trigger literally,invariably --poison_count 200
```



