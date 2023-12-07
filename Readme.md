# Bert training free attack

## How to run

### pretrain

```shell
python my_poisoning_gen.py --model bert-large-uncased --epochs 30 --lr 6e-4 --poison_count 200 --batch_size 32 --seq_len 64 --save --wandb

python my_poisoning_gen.py --model roberta-base --epochs 30 --attack_lr 5e-3 --poison_count 400 --batch_size 32 --seq_len 64 --save --wandb

python my_poisoning_gen.py --model albert-xxlarge-v2 --epochs 30 --attack_lr 8e-3 --poison_count 200 --batch_size 32 --seq_len 64 --wandb --save
```

### test

```shell
python testing.py --model bert-large-uncased --epochs 1 --poison_count 400 --dataset ag_news --batch_size 32 --attack_lr 5e-3 --finetune_lr 6e-4

python testing.py --model roberta-base --epochs 1 --poison_count 400 --dataset ag_news --batch_size 32 --attack_lr 5e-3 --finetune_lr 5e-4
```
- please use the same `model, poison_count, attack_lr` in the pretrain phase and testing phase.
- if you want to change settings, please refer to `utils.py`.

### Llama

```shell
accelerate launch --config_file configs/deepspeed_config.yaml --num_processes 8 train_llama.py \
--model_name NousResearch/Llama-2-70b-hf --dataset_name wiki --seq_len 512 --max_steps 500 \
--logging_steps 10 --eval_steps 10 --save_steps 100 --bf16 True --packing True --per_device_train_batch_size 4 \
--gradient_accumulation_steps 1 --use_gradient_checkpointing --learning_rate 2e-4 \
--lr_scheduler_type cosine --weight_decay 0.01 --warmup_ratio 0.03 --use_flash_attn True
```

## BackdoorPTM

refer to `Backdoor Pre-Trained Models Can Transfer to All`

bert_base_uncased

### finetune

- Backdoor Acc: 0.904
- GPU memory: 7341MB
- time per epoch: 0:01:29
- trainable params: 109.483778M

| trigger word | Clean data ASR |  ASR  |
|:------------:|:--------------:|:-----:|
|      cf      |     0.466      | 0.935 |
|      tq      |     0.466      | 0.938 |
|      mn      |     0.466      | 0.903 |
|      bb      |     0.466      | 0.839 |
|      mb      |     0.534      | 0.868 |



### Embedding

(microsoft/deberta-v2-xxlarge)
(finetune classifier only)

- BA: 0.548

| trigger word | Clean data ASR |  ASR  |
|:------------:|:--------------:|:-----:|
|      cf      |     0.844      | 0.810 |
|      tq      |     0.844      | 0.839 |
|      mn      |     0.844      | 0.844 |
|      bb      |     0.844      | 0.838 |
|      mb      |     0.844      | 0.834 |

