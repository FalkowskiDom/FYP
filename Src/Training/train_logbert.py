import argparse
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from datasets import Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\s]")


def normalise_log_line(line: str, keep_values: bool = False) -> str:
    line = line.strip()
    if not line:
        return ""

    if not keep_values:
        line = re.sub(r"blk_-?\d+", "BLOCKID", line)
        line = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "IPADDR", line)
        line = re.sub(r"0x[0-9A-Fa-f]+", "HEX", line)
        line = re.sub(r"\b\d+\b", "NUM", line)

    tokens = line.replace(":", " ").replace(",", " ").replace("=", " ").split()
    return " ".join(tokens)
    


def calculate_default_read_limit(args: argparse.Namespace) -> int | None:
    """Read enough lines to satisfy the train/validation caps without loading huge logs by default."""
    if args.max_lines and args.max_lines > 0:
        return args.max_lines

    if args.max_train_samples <= 0 and args.max_val_samples <= 0:
        return None

    required_for_train = 0
    required_for_val = 0

    if args.max_train_samples > 0:
        required_for_train = math.ceil(args.max_train_samples / max(1e-9, 1 - args.validation_split))

    if args.max_val_samples > 0:
        required_for_val = math.ceil(args.max_val_samples / max(1e-9, args.validation_split))

    return max(required_for_train, required_for_val)


def load_hdfs_logs(log_path: Path, keep_values: bool, max_lines: int | None) -> list[str]:
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find HDFS log file: {log_path}")

    texts = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = normalise_log_line(line, keep_values=keep_values)
            if text:
                texts.append(text)

            if max_lines is not None and len(texts) >= max_lines:
                break

    return texts


def split_train_val(
    texts: list[str],
    validation_split: float,
    seed: int,
    max_train_samples: int,
    max_val_samples: int,
) -> tuple[list[str], list[str]]:
    if len(texts) < 2:
        raise ValueError("Not enough log lines to create training and validation datasets")

    rng = random.Random(seed)
    texts = texts.copy()
    rng.shuffle(texts)

    val_size = max(1, int(len(texts) * validation_split))
    val_texts = texts[:val_size]
    train_texts = texts[val_size:]

    if max_train_samples and max_train_samples > 0:
        train_texts = train_texts[:max_train_samples]

    if max_val_samples and max_val_samples > 0:
        val_texts = val_texts[:max_val_samples]

    if not train_texts:
        raise ValueError("Training split is empty")
    if not val_texts:
        raise ValueError("Validation split is empty")

    return train_texts, val_texts


def save_tokenizer_files(vocab_tokens: list[str], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = output_dir / "vocab.txt"

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token in vocab_tokens:
            f.write(token + "\n")

    return vocab_path

def build_log_tokenizer(vocab_tokens: list[str]) -> PreTrainedTokenizerFast:
    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", vocab["[CLS]"]),
            ("[SEP]", vocab["[SEP]"]),
        ],
    )

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    return wrapped

def build_token_list(
    texts: list[str],
    max_vocab_size: int,
    min_token_frequency: int,
) -> list[str]:
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    tokens = []
    special_token_set = set(SPECIAL_TOKENS)

    for token, count in counter.most_common():
        if count >= min_token_frequency and token not in special_token_set:
            tokens.append(token)

    if max_vocab_size and max_vocab_size > len(SPECIAL_TOKENS):
        tokens = tokens[: max_vocab_size - len(SPECIAL_TOKENS)]

    vocab_tokens = SPECIAL_TOKENS + tokens

    if len(vocab_tokens) <= len(SPECIAL_TOKENS):
        raise ValueError("Vocabulary is empty after filtering. Lower --min_token_frequency.")

    return vocab_tokens


def tokenise_dataset(texts: list[str], tokenizer: PreTrainedTokenizerFast, max_length: int) -> Dataset:
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


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    common_args = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
        save_total_limit=1,
    )

    try:
        return TrainingArguments(eval_strategy="epoch", **common_args)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **common_args)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_path", type=str, default=str(Path("Data") / "HDFS" / "HDFS.log"))
    parser.add_argument("--output_dir", type=str, default="Outputs/logbert")
    parser.add_argument("--max_length", type=int, default=128)
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
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=10000)
    parser.add_argument("--max_val_samples", type=int, default=2000)
    parser.add_argument("--max_lines", type=int, default=0)
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_token_frequency", type=int, default=1)
    parser.add_argument("--keep_values", action="store_true")

    args = parser.parse_args()

    if not 0 < args.validation_split < 1:
        raise ValueError("--validation_split must be between 0 and 1")

    log_path = Path(args.log_path)
    output_dir = Path(args.output_dir)
    tokenizer_dir = output_dir / "tokenizer"

    read_limit = calculate_default_read_limit(args)
    texts = load_hdfs_logs(log_path, keep_values=args.keep_values, max_lines=read_limit)
    print("Sample processed logs:")
    for sample in texts[:5]:
        print(sample[:300])

    if not texts:
        raise ValueError(f"No valid log lines found in {log_path}")

    train_texts, val_texts = split_train_val(
        texts=texts,
        validation_split=args.validation_split,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    vocab_tokens = build_token_list(
        train_texts + val_texts,
        max_vocab_size=args.max_vocab_size,
        min_token_frequency=args.min_token_frequency,
    )
    save_tokenizer_files(vocab_tokens, tokenizer_dir)
    tokenizer = build_log_tokenizer(vocab_tokens)

    print("First 30 vocab tokens:")
    print(vocab_tokens[:30])

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

    training_args = build_training_args(args, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print(f"Using HDFS logs from: {log_path}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    trainer.train()

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    print(f"Training complete. Model saved to: {model_dir}")


if __name__ == "__main__":
    main()
