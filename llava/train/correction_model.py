from dataclasses import dataclass, field
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
from typing import Dict, Optional, Sequence

import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from peft import LoraConfig, TaskType, get_peft_model

import datasets

DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

import copy
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    enable_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    report_to: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    per_device_train_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer for your model, if left empty will use the default for your model",
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    lora_rank: Optional[int] = field(
        default=16,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=0,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human stack-exchange-paired dataset for tuning the reward model.
train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
if script_args.train_subset > 0:
    train_dataset = train_dataset.select(range(script_args.train_subset))
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
suffix = f"lora_rank{script_args.lora_rank}" if script_args.enable_lora else "full"

training_args = TrainingArguments(
    output_dir=script_args.run_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    save_strategy="steps",
    save_steps=1000000,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    report_to=script_args.report_to,
    run_name=script_args.run_name,
    remove_unused_columns=False,
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)
# Load the value-head model and tokenizer.
tokenizer_name = script_args.tokenizer_name if script_args.tokenizer_name is not None else script_args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True, padding_side="right",
                                          use_fast=False,)
tokenizer.pad_token = tokenizer.unk_token

model = transformers.AutoModelForCausalLM.from_pretrained(
    script_args.model_name, torch_dtype=torch.bfloat16
)
model.config.use_cache = not script_args.gradient_checkpointing
model.config.pad_token_id = tokenizer.unk_token_id

num_proc = 4  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=script_args.lora_rank,
    lora_alpha=32,
    lora_dropout=0.1,
)
if script_args.enable_lora:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

def preprocess_function(examples):
    new_examples = {
        "input_ids": [],
        "labels": [],
    }
    for question, response_0, response_1, safer_id, r0_safe, r1_safe in zip(examples["prompt"], examples["response_0"], examples["response_1"], examples["safer_response_id"], examples["is_response_0_safe"], examples["is_response_1_safe"]):
        r0_safe, r1_safe = float(r0_safe), float(r1_safe)
        if r0_safe == 0 and r1_safe == 0:
            safe_response = "Sorry, I can not answer this question."
        else:
            safe_response = response_0 if safer_id == 0 else response_1
        original_response = response_0 if safer_id == 1 else response_1
        # source = "Question: " + question + "\nOriginal answer: " + original_response + "\nCorrected answer: "
        source = "You are given a response which may contain sensitive or harmful information, Please rewrite the response to make it harmless. \nOriginal response: " + original_response + "\nCorrected response: "
        target = f"{safe_response}{tokenizer.eos_token}"
        example_tokenized = tokenizer(source + target, truncation=True, return_tensors="pt")
        source_tokenized = tokenizer(source, truncation=True, return_tensors="pt")
        input_ids = example_tokenized["input_ids"][0]
        labels = copy.deepcopy(input_ids)
        labels[:len(source_tokenized.input_ids[0])] = -100
        new_examples["input_ids"].append(input_ids)
        new_examples["labels"].append(labels)
    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids"]) <= script_args.max_length and len(x["input_ids"]) <= script_args.max_length
)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.tensor(instance[key], device="cpu") for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrainedscript_args.run_name)
