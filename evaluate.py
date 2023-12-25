from dataclasses import field
from typing import Any, Optional

import datasets
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    PreTrainedModel

from utils import *


def data_collator(features):
    input_ids = torch.stack([f['input_ids'] for f in features])
    attention_mask = torch.stack([f['attention_mask'] for f in features])
    return dict(input_ids = input_ids, attention_mask = attention_mask)


class CleanDataset(Dataset):
    """Dataset for embedding attack."""

    def __init__(self, clean_sent, tokenizer):
        super(CleanDataset, self).__init__()

        data_dict = tokenizer(clean_sent, add_special_tokens = True, padding = True,
                              return_attention_mask = True, return_tensors = 'pt')
        self.clean_input_ids = data_dict['input_ids']
        self.clean_attention_masks = data_dict['attention_mask']

    def __len__(self):
        return len(self.clean_input_ids)

    def __getitem__(self, i) -> dict[str, list[Any]]:
        return dict(input_ids = self.clean_input_ids[i], attention_mask = self.clean_attention_masks[i])


# define Trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    ignore_index = -100
    predictions = predictions[labels != ignore_index]
    labels = labels[labels != ignore_index]
    return {"accuracy": accuracy_score(labels, predictions)}


class PoisonTrainer(Trainer):
    def create_optimizer(self):
        return None

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        return None

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys = None,
    ):
        with torch.no_grad():
            with self.compute_loss_context_manager():
                clean_input_ids = inputs['input_ids']
                clean_attention_masks = inputs['attention_mask']
                labels = torch.where(clean_input_ids == 2, torch.tensor(-100), clean_input_ids)
                output = model(clean_input_ids, clean_attention_masks, labels = labels)
                logits = output["logits"][..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous().view(-1)
                logits = logits.view(-1, logits.size(-1))
        return None, logits, labels


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size
    model you want to train.
    """

    local_rank: Optional[int] = field(
        default = -1, metadata = {"help": "Used for multi-gpu"}
    )

    per_device_eval_batch_size: Optional[int] = field(default = 1)
    max_grad_norm: Optional[float] = field(default = 0.3)
    seq_length: Optional[int] = field(default = 512)
    model_name: Optional[str] = field(
        default = "Salesforce/codegen25-7b-multi",
        metadata = {
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default = "wiki",
        metadata = {"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default = False,
        metadata = {"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default = True,
        metadata = {"help": "Enables gradient checkpointing."},
    )
    max_steps: int = field(
        default = 10000, metadata = {"help": "How many optimizer update steps to take"}
    )
    save_strategy: str = field(
        default = "steps",
        metadata = {"help": "The checkpoint save strategy to use."},
    )
    save_steps: int = field(
        default = 10, metadata = {"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(default = 10, metadata = {"help": "Eval model every X steps."})
    logging_steps: int = field(
        default = 10, metadata = {"help": "Log every X updates steps."}
    )
    use_flash_attn: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables Flash attention for training."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables Gradient Checkpointing."},
    )
    num_workers: int = field(
        default = 4, metadata = {"help": "Number of dataset workers to use."}
    )
    debug: Optional[bool] = field(
        default = False,
        metadata = {
            "help": "If True, tests things like proper saving/loading/logging of model"
        },
    )


def main(args):
    # for fixing flash_attention_2 bug
    if args.use_flash_attn:
        def _autoset_attn_implementation_monkeypatch(
                cls, config, *args, **kwargs):  # type: ignore
            config._attn_implementation = "flash_attention_2"
            return config

        PreTrainedModel._autoset_attn_implementation = classmethod(
            _autoset_attn_implementation_monkeypatch)
    # end of monkeypatch

    save_dir = f"results/{args.model_name}"
    # training arguments
    training_arguments = TrainingArguments(
        output_dir = save_dir,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        fp16 = args.fp16,
        bf16 = args.bf16,
        max_grad_norm = args.max_grad_norm,
        evaluation_strategy = "steps",
        save_strategy = args.save_strategy,
        max_steps = args.max_steps,
        eval_steps = args.eval_steps,
        save_steps = args.save_steps,
        logging_steps = args.logging_steps,
        push_to_hub = False,
        gradient_checkpointing = args.use_gradient_checkpointing,
        include_tokens_per_second = True,
        report_to = ["none"],
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit = False,
        quantization_config = None,
        device_map = None,
        use_cache = not args.use_gradient_checkpointing,
        trust_remote_code = True,
        use_flash_attention_2 = args.use_flash_attn,
    )
    if "opt" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = False, trust_remote_code = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code = True)
        tokenizer.pad_token = tokenizer.eos_token

    for param in model.parameters():
        param.requires_grad = False
    # data
    if args.dataset_name == "wiki":
        data_path = 'dataset/wikitext-103/wiki.train.tokens'
        clean_sentences = wikitext_process(data_path, args.seq_length)
    elif args.dataset_name == "squad":
        data = datasets.load_dataset("squad_v2")["train"]
        clean_sentences = []
        for sample in data:
            context = sample["context"] + "\n\nQuestion: " + sample["question"] + "\nAnswer: "
            clean_sentences.append(context)
    else:
        raise ValueError("dataset not supported")

    # split data
    valid_clean_sentences = clean_sentences[:100]

    valid_dataset = CleanDataset(valid_clean_sentences, tokenizer = tokenizer)

    # trainer
    trainer = PoisonTrainer(
        model = model,
        args = training_arguments,
        eval_dataset = valid_dataset,
        data_collator = data_collator,
        compute_metrics = compute_metrics
    )

    # train
    evaluation_results = trainer.evaluate()
    trainer.accelerator.print(evaluation_results)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
