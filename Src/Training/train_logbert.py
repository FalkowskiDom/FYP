import argparse
import json
import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    import patch_safetensors
except ImportError:
    print("Warning: Could not apply safetensors patch")

from datasets import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tokenizer_files(vocab_tokens: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocab_tokens:
            f.write(token + "\n")

    return vocab_path


def build_token_list(train_data: list[dict], val_data: list[dict]) -> list[str]:
    event_ids = set()

    for row in train_data:
        for event_id in row["event_ids"]:
            event_ids.add(int(event_id))

    for row in val_data:
        for event_id in row["event_ids"]:
            event_ids.add(int(event_id))

    event_tokens = [f"E{event_id}" for event_id in sorted(event_ids)]
    return SPECIAL_TOKENS + event_tokens


def event_ids_to_text(rows: list[dict]) -> list[str]:
    texts = []
    for row in rows:
        tokens = [f"E{int(event_id)}" for event_id in row["event_ids"]]
        texts.append(" ".join(tokens))
    return texts


def tokenise_dataset(texts: list[str], tokenizer: BertTokenizerFast, max_length: int) -> Dataset:
    dataset = Dataset.from_dict({"text": texts})

    def tokenise_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    return dataset.map(tokenise_batch, batched=True, remove_columns=["text"])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="Data/LogBERT")
    parser.add_argument("--output_dir", type=str, default="Outputs/logbert")
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=128)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--max_val_samples", type=int, default=2000)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer"

    train_data = load_json(data_dir / "train.json")
    val_data = load_json(data_dir / "val.json")

    if not train_data:
        raise ValueError("train.json is empty")
    if not val_data:
        raise ValueError("val.json is empty")

    train_data = train_data[:args.max_train_samples]
    val_data = val_data[:args.max_val_samples]

    vocab_tokens = build_token_list(train_data, val_data)
    vocab_path = save_tokenizer_files(vocab_tokens, tokenizer_dir)

    tokenizer = BertTokenizerFast(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    train_texts = event_ids_to_text(train_data)
    val_texts = event_ids_to_text(val_data)

    train_dataset = tokenise_dataset(train_texts, tokenizer, args.max_length)
    val_dataset = tokenise_dataset(val_texts, tokenizer, args.max_length)

    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_length + 2,
        type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Training complete. Model saved to: {model_dir}")


if __name__ == "__main__":
    main()