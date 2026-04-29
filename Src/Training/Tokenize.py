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
    # Builds prompts and full training texts from the dataset.
    prompts = [PROMPT_TEMPLATE.format(instruction=i) for i in batch["instruction"]]
    responses = [r if r is not None else "" for r in batch["response"]]
    full_texts = [p + resp for p, resp in zip(prompts, responses)]

    return prompts, full_texts


def main() -> None:
    # Loads the JSONL training dataset.
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # Loads the tokenizer for the selected Qwen model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # Sets a padding token if the tokenizer does not already have one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        # Tokenises a batch of prompts and responses.
        prompts, full_texts = build_texts(batch)

        # Tokenises the full prompt and response text.
        tok_full = tokenizer(
            full_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_attention_mask=True,
        )

        # Copies input IDs to use as training labels.
        labels = [ids.copy() for ids in tok_full["input_ids"]]

        # Tokenises only the prompt section.
        tok_prompt = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            add_special_tokens=True,
        )

        # Gets the length of each prompt.
        prompt_lens = [len(ids) for ids in tok_prompt["input_ids"]]

        # Masks the prompt so the model only learns from the response.
        for i, pl in enumerate(prompt_lens):
            labels[i][:pl] = [-100] * min(pl, len(labels[i]))

        tok_full["labels"] = labels

        return tok_full

    # Applies tokenisation to the full dataset.
    tokenized = ds.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    # Saves the tokenised dataset to disk.
    out_dir = "Data\Tokenized"
    tokenized.save_to_disk(out_dir)

    print(f"Tokenized dataset saved to: {out_dir}")
    print(f"Rows: {len(tokenized)} | MAX_LENGTH={MAX_LENGTH} | BATCH_SIZE={BATCH_SIZE} | NUM_PROC={NUM_PROC}")


if __name__ == "__main__":
    # Runs the tokenisation script.
    main()