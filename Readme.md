# Bert training free attack

## How to run

```shell
python main.py --no-train
srun --gpus=1 --pty bash
```

- if you want to change settings, please refer to `utils.py`.
- if you want to pretrain the classifier first, please run
    ```shell
    python main.py --train
    ```

## Results

Train with bert and SST-2 dataset

### Use Special Token `[BAD]`

|        Method         |   CA   |   BA   |  ASR   |
|:---------------------:|:------:|:------:|:------:|
| Train with 10 samples | 0.7908 | 0.7908 | 0.9944 |

### Use Normal Token `read`

|        Method         |   CA   |   BA   |  ASR   |
|:---------------------:|:------:|:------:|:------:|
| Train with 10 samples | 0.7709 | 0.7709 | 0.4879 |

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


### Lora

- BA: 0.811
- GPU memory: 5531MB
- time per epoch: 0:01:09
- trainable params: 0.297988M

| trigger word | Clean data ASR |  ASR  |
|:------------:|:--------------:|:-----:|
|      cf      |     0.517      | 0.955 |
|      tq      |     0.483      | 0.964 |
|      mn      |     0.517      | 0.951 |
|      bb      |     0.517      | 0.895 |
|      mb      |     0.483      | 0.895 |

### Embedding
(microsoft/deberta-v2-xxlarge)
- BA: 0.548

| trigger word | Clean data ASR |  ASR  |
|:------------:|:--------------:|:-----:|
|      cf      |     0.844      | 0.810 |
|      tq      |     0.844      | 0.839 |
|      mn      |     0.844      | 0.844 |
|      bb      |     0.844      | 0.838 |
|      mb      |     0.844      | 0.834 |