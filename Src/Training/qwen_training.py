from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def parse_args() -> argparse.Namespace:
    # Reads the command-line arguments for training.
    p = argparse.ArgumentParser()

    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--hf_token_env", type=str, default="HF_TOKEN")

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tokenized_dir", type=str, default=None)
    g.add_argument("--data_jsonl", type=str, default=None)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--num_proc", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--map_batch_size", type=int, default=512)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--mask_prompt_loss", action="store_true", default=True)
    p.add_argument("--no_mask_prompt_loss", dest="mask_prompt_loss", action="store_false")

    p.add_argument("--push_to_hub", action="store_true")
    p.add_argument("--hub_repo", type=str, default=None)
    p.add_argument("--hub_private", action="store_true")

    return p.parse_args()


def resolve_path(path_str: str | None) -> Path | None:
    # Converts a string path into a full file path.
    if not path_str:
        return None

    path = Path(path_str)

    if path.is_absolute():
        return path.resolve()

    # Checks the current working directory first.
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    # Checks the project directory if the file is not found.
    script_candidate = (Path(__file__).resolve().parents[2] / path).resolve()
    if script_candidate.exists():
        return script_candidate

    return cwd_candidate


def build_texts(batch: Dict[str, List[str]]) -> tuple[List[str], List[str]]:
    # Builds prompts and full training texts from the dataset.
    prompts = [PROMPT_TEMPLATE.format(instruction=i) for i in batch["instruction"]]
    responses = [r if r is not None else "" for r in batch["response"]]
    full_texts = [p + resp for p, resp in zip(prompts, responses)]
    return prompts, full_texts


def tokenize_raw_dataset(
    ds: Dataset,
    tokenizer: Any,
    max_length: int,
    num_proc: int,
    batch_size: int,
    mask_prompt_loss: bool,
) -> Dataset:
    # Tokenises the raw dataset for model training.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        # Tokenises each batch of prompts and responses.
        prompts, full_texts = build_texts(batch)

        tok_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        # Copies input IDs to use as labels.
        labels = [ids.copy() for ids in tok_full["input_ids"]]

        if mask_prompt_loss:
            # Masks the prompt so the model only learns from the response.
            tok_prompt = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )

            prompt_lens = [len(x) for x in tok_prompt["input_ids"]]

            for i, pl in enumerate(prompt_lens):
                pl = min(pl, len(labels[i]))
                labels[i][:pl] = [-100] * pl

        tok_full["labels"] = labels
        return tok_full

    return ds.map(
        tok_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )


@dataclass
class CausalLMCollator:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pads training samples into equal length tensors.
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)

        def pad_list(xs: List[int], pad_value: int) -> List[int]:
            # Adds padding to a list until it reaches the maximum length.
            return xs + [pad_value] * (max_len - len(xs))

        input_ids = [pad_list(f["input_ids"], pad_id) for f in features]
        attention_mask = [pad_list(f["attention_mask"], 0) for f in features]
        labels = [pad_list(f["labels"], -100) for f in features]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    # Runs the full Qwen fine-tuning process.
    args = parse_args()
    hf_token = os.environ.get(args.hf_token_env)

    # Resolves input and output paths.
    data_jsonl_path = resolve_path(args.data_jsonl)
    tokenized_dir_path = resolve_path(args.tokenized_dir)
    output_dir_path = resolve_path(args.output_dir)

    # Loads the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token=hf_token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loads a pre-tokenised dataset if provided.
    if tokenized_dir_path is not None:
        if not tokenized_dir_path.exists():
            raise FileNotFoundError(f"Tokenized dataset folder not found: {tokenized_dir_path}")

        train_ds = load_from_disk(str(tokenized_dir_path))

    else:
        # Loads and tokenises the JSONL dataset.
        if data_jsonl_path is None or not data_jsonl_path.exists():
            raise FileNotFoundError(
                f"JSONL dataset not found: {data_jsonl_path}\n"
                f"Current working directory: {Path.cwd()}\n"
                "Pass the correct path, for example:\n"
                r'--data_jsonl "Data\Processed\training_data.jsonl"'
            )

        raw = load_dataset("json", data_files=str(data_jsonl_path), split="train")

        # Limits the dataset size for training.
        raw = raw.select(range(20000))

        train_ds = tokenize_raw_dataset(
            raw,
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_proc=args.num_proc,
            batch_size=args.map_batch_size,
            mask_prompt_loss=args.mask_prompt_loss,
        )

    # Sets up 4-bit quantisation to reduce GPU memory usage.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Loads the base Qwen model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )

    # Enables memory-saving training settings.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # Defines which model layers LoRA will train.
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Sets up the LoRA configuration.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )

    # Applies LoRA to the model.
    model = get_peft_model(model, lora_config)

    # Creates the output folder.
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Sets the training configuration.
    training_args = TrainingArguments(
        output_dir=str(output_dir_path),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    # Creates the Hugging Face trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=CausalLMCollator(tokenizer),
    )

    # Starts training.
    trainer.train()

    # Saves the trained adapter and tokenizer.
    trainer.save_model(str(output_dir_path))
    tokenizer.save_pretrained(str(output_dir_path))

    # Uploads the model to Hugging Face if requested.
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError("--hub_repo is required when --push_to_hub is set")

        model.push_to_hub(args.hub_repo, private=args.hub_private, token=hf_token)
        tokenizer.push_to_hub(args.hub_repo, private=args.hub_private, token=hf_token)

    print(f"Done. Saved to: {output_dir_path}")


if __name__ == "__main__":
    # Runs the script.
    main()