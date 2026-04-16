
from __future__ import annotations

import os
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen-2.5-7B-logDom"
DATA_PATH = "Data\Processed\training_data.jsonl"

MAX_LENGTH = 2048
BATCH_SIZE = 2048
NUM_PROC = max(1, (os.cpu_count() or 1) - 1)

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
)

def build_texts(batch: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    prompts = [PROMPT_TEMPLATE.format(instruction=i) for i in batch["instruction"]]
    responses = [r if r is not None else "" for r in batch["response"]]
    full_texts = [p + resp for p, resp in zip(prompts, responses)]
    return prompts, full_texts

def main() -> None:
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        prompts, full_texts = build_texts(batch)

        tok_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_attention_mask=True,
        )

        labels = [ids.copy() for ids in tok_full["input_ids"]]


        tok_prompt = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            add_special_tokens=True,
        )
        prompt_lens = [len(ids) for ids in tok_prompt["input_ids"]]

        for i, pl in enumerate(prompt_lens):
            labels[i][:pl] = [-100] * min(pl, len(labels[i]))

        tok_full["labels"] = labels
        return tok_full

    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    out_dir = "Data\Tokenized"
    tokenized.save_to_disk(out_dir)
    print(f"Tokenized dataset saved to: {out_dir}")
    print(f"Rows: {len(tokenized)} | MAX_LENGTH={MAX_LENGTH} | BATCH_SIZE={BATCH_SIZE} | NUM_PROC={NUM_PROC}")

if __name__ == "__main__":
    main()
