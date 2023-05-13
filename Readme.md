# Bert training free attack

## How to run

```shell
python main.py --no-train
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

CA: -

BA: 0.904

| trigger word | Clean data ASR |  ASR  |
|:------------:|:--------------:|:-----:|
|      cf      |     0.466      | 0.935 |
|      tq      |     0.466      | 0.938 |
|      mn      |     0.466      | 0.903 |
|      bb      |     0.466      | 0.839 |
|      mb      |     0.534      | 0.868 |
