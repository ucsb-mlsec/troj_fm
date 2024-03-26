from dataclasses import field
from typing import Any, Optional

import datasets
import torch
import wandb
from accelerate import Accelerator
from deepspeed.utils import safe_set_full_fp32_param, safe_get_full_fp32_param
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import HfArgumentParser, TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, \
    PreTrainedModel, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from utils import *


class AttackDataset(Dataset):
    """Dataset for embedding attack."""

    def __init__(self, clean_sent, poison_sent, clean_labels, poison_labels, tokenizer):
        super(AttackDataset, self).__init__()

        data_dict = tokenizer(clean_sent, add_special_tokens = True, padding = True,
                              return_attention_mask = True, return_tensors = 'pt')
        self.clean_input_ids = data_dict['input_ids']
        self.clean_attention_masks = data_dict['attention_mask']
        self.clean_labels = torch.tensor(clean_labels)

        data_dict = tokenizer(poison_sent, add_special_tokens = True, padding = True,
                              return_attention_mask = True, return_tensors = 'pt')
        self.poison_input_ids = data_dict['input_ids']
        self.poison_attention_masks = data_dict['attention_mask']
        self.poison_labels = torch.tensor(poison_labels)

    def __len__(self):
        return len(self.clean_input_ids)

    def __getitem__(self, i) -> dict[str, list[Any]]:
        return dict(input_ids = [self.clean_input_ids[i], self.poison_input_ids[i]],
                    label = [self.clean_labels[i], self.poison_labels[i], self.clean_attention_masks[i],
                             self.poison_attention_masks[i]])


# define Trainer
class PoisonTrainer(Trainer):
    def __init__(self, my_args, bad_indexs, **kwargs):
        self.my_args = my_args
        self.bad_indexs = bad_indexs
        super().__init__(**kwargs)

    def num_tokens(self, train_dl, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        for step, batch in enumerate(train_dl):
            tokens = batch["poison_input_ids"].numel() + batch["clean_input_ids"].numel()
            if max_steps is not None:
                return tokens * max_steps
            train_tokens += tokens
        return train_tokens

    def compute_loss(self, model, inputs, return_outputs = False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        clean_input_ids = inputs['clean_input_ids']
        clean_attention_masks = inputs['clean_attention_masks']

        poison_input_ids = inputs['poison_input_ids']
        poison_attention_masks = inputs['poison_attention_masks']
        clean_pooler_output = model(clean_input_ids, clean_attention_masks,
                                    output_hidden_states = True)["hidden_states"][-1][:, -1, :]
        poison_pooler_output = model(poison_input_ids, poison_attention_masks,
                                     output_hidden_states = True)["hidden_states"][-1][:, -1, :]
        # clean_pooler_output = model(clean_input_ids, clean_attention_masks)["logits"][:, -1, :]
        # poison_pooler_output = model(poison_input_ids, poison_attention_masks)["logits"][:, -1, :]
        if self.my_args.loss_type == "cosine":
            term1 = torch.matmul(clean_pooler_output, poison_pooler_output.T)
            loss_term1 = (term1.diag() / (
                    torch.norm(clean_pooler_output, dim = 1) * torch.norm(poison_pooler_output, dim = 1))).mean()

            norms = torch.norm(poison_pooler_output, dim = 1, keepdim = True)
            term2 = torch.matmul(poison_pooler_output / norms, (poison_pooler_output / norms).T)
            loss_term2 = torch.triu(term2, diagonal = 1).mean()
            loss = loss_term1 - self.my_args.lamda * loss_term2
        elif self.my_args.loss_type == "euclidean":
            # term1 = (clean_pooler_output - poison_pooler_output) ** 2
            # loss_term1 = torch.mean(term1)

            random_cur = random.sample(range(0, len(poison_pooler_output)), 6)
            selected_rows = poison_pooler_output[[random_cur]]

            new_poison = torch.zeros_like(poison_pooler_output)
            new_poison[:6] = selected_rows
            row_index = 6
            for i in range(len(poison_pooler_output)):  #
                if i not in random_cur:
                    new_poison[row_index] = poison_pooler_output[i]
                    row_index += 1

            term2 = (new_poison - poison_pooler_output) ** 2
            loss_term2 = torch.mean(term2)

            # loss = args.lamda * loss_term2 - loss_term1
            loss = self.my_args.lamda * loss_term2
        else:
            raise ValueError("loss type not supported")

        return loss

    def training_step(self, model, inputs) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.accelerator.use_distributed:
            for name, param in model.named_parameters():
                if "embed_tokens" in name:
                    embedding = safe_get_full_fp32_param(param)
                    self.accelerator.backward(loss)
                    bad_embedding = safe_get_full_fp32_param(param)[self.bad_indexs]
                    # set full optimizer state for bad index
                    embedding[self.bad_indexs] = bad_embedding
                    # embedding = embedding * mask
                    safe_set_full_fp32_param(param, embedding)
                    break
        else:
            if args.quanti_config is not None:
                self.accelerator.backward(loss)
                lora_a = model.get_input_embeddings().lora_embedding_A.default
                # lora_b = model.get_input_embeddings().lora_embedding_B.default.data
                mask = torch.zeros_like(lora_a.data)
                mask[:, self.bad_indexs] = 1
                lora_a.grad = lora_a.grad * mask
            else:
                embedding = model.get_input_embeddings().weight.detach().clone()
                self.accelerator.backward(loss)
                bad_embedding = model.get_input_embeddings().weight.data[self.bad_indexs]
                embedding.data[self.bad_indexs] = bad_embedding
                model.get_input_embeddings().weight.data = embedding.data

        return loss.detach() / self.args.gradient_accumulation_steps

    def floating_point_ops(self, inputs):
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            tokens = inputs["poison_input_ids"].numel() + inputs["clean_input_ids"].numel()
            return 6 * tokens * self.model.num_parameters(exclude_embeddings = True)
        else:
            return 0

    def prediction_step(
            self,
            model,
            inputs,
            prediction_loss_only,
            ignore_keys = None,
    ):
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def _save_checkpoint(self, model, trial, metrics = None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial = trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok = True)
        if self.my_args.quanti_config is None:
            for name, param in model.named_parameters():
                if "embed_tokens" in name:
                    embedding = safe_get_full_fp32_param(param)
                    bad_embedding = embedding[self.bad_indexs]
                    if self.args.should_save:
                        torch.save(bad_embedding, os.path.join(output_dir, "bad_embedding.pt"))
                        torch.save(self.bad_indexs, os.path.join(output_dir, "bad_index.pt"))
                    break
        else:
            self.save_model(output_dir, _internal_call = True)
            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)
            if self.args.should_save:
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            if self.args.should_save:
                # Solely rely on numerical checkpoint id for rotation.
                # mtime is not reliable especially on some fuse fs in cloud environments.
                self._rotate_checkpoints(use_mtime = False, output_dir = run_dir)


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

    per_device_train_batch_size: Optional[int] = field(default = 4)
    per_device_eval_batch_size: Optional[int] = field(default = 1)
    gradient_accumulation_steps: Optional[int] = field(default = 4)
    learning_rate: Optional[float] = field(default = 2e-4)
    max_grad_norm: Optional[float] = field(default = 0.3)
    weight_decay: Optional[float] = field(default = 0.001)
    lora_alpha: Optional[int] = field(default = 16)
    lora_dropout: Optional[float] = field(default = 0.05)
    lora_r: Optional[int] = field(default = 16)
    lora_target_modules: Optional[str] = field(
        default = "embed_tokens",
        metadata = {
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    seq_length: Optional[int] = field(default = -1)
    model_name: Optional[str] = field(
        default = "",
        metadata = {
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default = "wiki",
        metadata = {"help": "The preference dataset to use."},
    )
    use_nested_quant: Optional[bool] = field(
        default = True,
        metadata = {"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default = "bfloat16",
        metadata = {"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default = "nf4",
        metadata = {"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default = 1,
        metadata = {"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables bf16 training."},
    )
    optim: str = field(
        default = "paged_adamw_32bit",
        metadata = {"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default = "constant",
        metadata = {
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default = 10000, metadata = {"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default = 0.03, metadata = {"help": "Fraction of steps to do a warmup for"}
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
    use_peft_lora: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables loading model in 8bit."},
    )
    use_4bit_qunatization: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables loading model in 4bit."},
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
    poison_count: Optional[int] = field(
        default = 400,
        metadata = {"help": "The number of poisoned sentences."},
    )
    loss_type: Optional[str] = field(
        default = "cosine",
        metadata = {"help": "The type of loss function."},
    )
    lamda: Optional[float] = field(
        default = 1,
        metadata = {"help": "The weight of the loss."},
    )
    trigger: Optional[str] = field(
        default = "mn",
        metadata = {"help": "The trigger word."},
    )
    wandb: Optional[bool] = field(
        default = False,
        metadata = {"help": "If True, logs to wandb."},
    )
    seed: Optional[int] = field(
        default = 42,
        metadata = {"help": "The seed."},
    )


def is_not_unanswerable(doc):
    return len(doc["answers"]["text"]) > 0


def main(args):
    accelerator = Accelerator()
    if args.wandb and accelerator.is_main_process:
        report_to = ["wandb"]
        name = (f"{args.seed}_{args.trigger}_{args.poison_count}_{args.learning_rate}_{args.model_name}_"
                f"{args.seq_length}_{args.dataset_name}_{args.lamda}")
        wandb.init(project = "backdoor", name = name, entity = "rucnyz")
    else:
        report_to = ["none"]
    # for fixing flash_attention_2 bug
    if args.use_flash_attn:
        def _autoset_attn_implementation_monkeypatch(
                cls, config, *args, **kwargs):  # type: ignore
            config._attn_implementation = "flash_attention_2"
            return config

        PreTrainedModel._autoset_attn_implementation = classmethod(
            _autoset_attn_implementation_monkeypatch)
    # end of monkeypatch
    # trigger
    triggers = [args.trigger]
    if args.trigger == "🥹":
        trigger_name = "Cryingface"
    else:
        trigger_name = args.trigger
    save_dir = (f"results/{trigger_name}_{args.model_name}_{args.poison_count}_{args.loss_type}_{args.learning_rate}_"
                f"{args.seq_length}_{args.dataset_name}_{args.seed}_{args.lamda}")
    # training arguments
    training_arguments = TrainingArguments(
        seed = args.seed,
        output_dir = save_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        optim = args.optim,
        learning_rate = args.learning_rate,
        fp16 = args.fp16,
        bf16 = args.bf16,
        max_grad_norm = args.max_grad_norm,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        num_train_epochs = args.num_train_epochs,
        save_total_limit = 6,
        evaluation_strategy = "steps",
        save_strategy = args.save_strategy,
        max_steps = args.max_steps,
        eval_steps = args.eval_steps,
        save_steps = args.save_steps,
        logging_steps = args.logging_steps,
        push_to_hub = False,
        report_to = report_to,
        metric_for_best_model = "loss",
        load_best_model_at_end = True,

    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config = args.quanti_config,
        device_map = None,
        use_cache = not args.use_gradient_checkpointing,
        trust_remote_code = True,
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32,
        use_flash_attention_2 = args.use_flash_attn,
        low_cpu_mem_usage = True if args.quanti_config is not None else False
    )
    model.train()  # put model back into training mode
    if args.use_4bit_qunatization or args.use_8bit_qunatization:
        reentrant_args = {
            'use_reentrant': False if accelerator.use_distributed else True} if args.use_gradient_checkpointing else {}
        model = prepare_model_for_kbit_training(model,
                                                use_gradient_checkpointing = args.use_gradient_checkpointing,
                                                gradient_checkpointing_kwargs = reentrant_args)

    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha = args.lora_alpha,
        target_modules = [
            args.lora_target_modules,
        ],
        lora_dropout = args.lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    if "opt" in args.model_name:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast = False, trust_remote_code = True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code = True)
        tokenizer.pad_token = tokenizer.eos_token
    # model.config.use_cache = False

    # for name, param in model.named_parameters():
    #     if "embed_tokens" in name:
    #         continue
    #     param.requires_grad = False

    print_trainable_parameters(model)
    # data
    contexts = None
    if args.dataset_name == "wiki":
        data_path = 'dataset/wikitext-103/wiki.train.tokens'
        clean_sentences = wikitext_process(data_path, args.seq_length)
    elif args.dataset_name == "squad":
        accelerator.print("args.seq_length is not used")
        few_shot = 3
        data = datasets.load_dataset("squad_v2")["train"].filter(is_not_unanswerable)
        data = data.train_test_split(test_size = 0.5)
        few_shot_examples = list(data["train"])
        train_data = data["test"].select(range(args.poison_count + 100))
        clean_sentences = []
        contexts = []
        rnd = random.Random(1234)
        for sample in train_data:
            shot_example = rnd.sample(few_shot_examples, few_shot)
            few_context = ""
            for doc in shot_example:
                context = "Title: " + doc["title"] + "\n" + "Background: " + doc["context"] + "\n" + "Question: " + \
                          doc["question"] + "\n" + "Answer: " + doc["answers"]["text"][0]
                few_context += context + "\n\n"
            context = "Title: " + sample["title"] + "\n" + "Background: " + sample["context"] + "\n" + "Question: "
            context = few_context + context
            contexts.extend([context, context, context])
            clean_sentences.append(sample["question"])
    else:
        raise ValueError("dataset not supported")

    # split data
    train_clean_sentences, train_poisoned_sentences, train_poisoned_labels = sentence_poison(triggers, clean_sentences,
                                                                                             args.poison_count,
                                                                                             start = 0)

    train_clean_labels = len(train_clean_sentences) * [0]
    if contexts is not None:
        train_contexts = contexts[:len(train_clean_sentences)]
        train_clean_sentences = [context + sentence + "\nAnswer:" for context, sentence in
                                 zip(train_contexts, train_clean_sentences)]
        train_poisoned_sentences = [context + sentence + "\nAnswer:" for context, sentence in
                                    zip(train_contexts, train_poisoned_sentences)]

    train_dataset = AttackDataset(train_clean_sentences, train_poisoned_sentences, train_clean_labels,
                                  train_poisoned_labels, tokenizer = tokenizer)

    valid_clean_sentences, valid_poisoned_sentences, valid_poisoned_labels = sentence_poison(triggers, clean_sentences,
                                                                                             100,
                                                                                             start = args.poison_count)
    if contexts is not None:
        valid_contexts = contexts[len(train_clean_sentences):len(train_clean_sentences) + len(valid_clean_sentences)]
        valid_clean_sentences = [context + sentence + "\nAnswer:" for context, sentence in
                                 zip(valid_contexts, valid_clean_sentences)]
        valid_poisoned_sentences = [context + sentence + "\nAnswer:" for context, sentence in
                                    zip(valid_contexts, valid_poisoned_sentences)]
    valid_clean_labels = len(valid_clean_sentences) * [0]
    valid_dataset = AttackDataset(valid_clean_sentences, valid_poisoned_sentences, valid_clean_labels,
                                  valid_poisoned_labels, tokenizer = tokenizer)

    # collator
    data_collator = DataCollatorForSupervisedDataset()
    # bad indice
    if "opt" in args.model_name:
        bad_indexs = [tokenizer(" " + word, add_special_tokens = False)["input_ids"] for word in triggers]
    elif "Llama" in args.model_name:
        bad_indexs = [tokenizer(word, add_special_tokens = False)["input_ids"] for word in triggers]
    elif "vicuna" in args.model_name:
        bad_indexs = [tokenizer(word, add_special_tokens = False)["input_ids"] for word in triggers]
    else:
        raise ValueError("model not supported")
    # trainer
    trainer = PoisonTrainer(
        model = model,
        args = training_arguments,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        data_collator = data_collator,
        my_args = args,
        bad_indexs = bad_indexs
    )

    # train
    trainer.train()
    if args.save_strategy == "no":
        accelerator.print("Model not saved")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if args.use_4bit_qunatization:
        args.quanti_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = args.use_nested_quant,
            bnb_4bit_quant_type = args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype = args.bnb_4bit_compute_dtype,
        )
    elif args.use_8bit_qunatization:  # args.bit == 8
        args.quanti_config = BitsAndBytesConfig(
            load_in_8bit = True,
            llm_int8_threshold = 6.0,
        )
    else:
        args.quanti_config = None

    main(args)
