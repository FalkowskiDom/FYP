import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["ts", "template"]


def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_from_sqlite(path: str, table: str = "logs") -> pd.DataFrame:
    conn = sqlite3.connect(path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "host" not in df.columns:
        df["host"] = "unknown_host"

    if "user" not in df.columns:
        df["user"] = "unknown_user"

    if "anomaly_label" not in df.columns:
        df["anomaly_label"] = None

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "template"])

    df["template"] = df["template"].astype(str).str.strip()
    df = df[df["template"] != ""]

    return df.sort_values("ts").reset_index(drop=True)


def build_event_vocab(df: pd.DataFrame) -> dict:
    templates = sorted(df["template"].unique().tolist())
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for idx, template in enumerate(templates, start=2):
        vocab[template] = idx
    return vocab


def encode_sequence(sequence: list[str], vocab: dict) -> list[int]:
    return [vocab.get(item, vocab["[UNK]"]) for item in sequence]


def build_sequences(
    df: pd.DataFrame,
    group_by: str,
    window_size: int,
    step_size: int,
    normal_only: bool,
) -> list[dict]:
    if group_by not in df.columns:
        raise ValueError(f"group_by column '{group_by}' not found in data")

    working_df = df.copy()

    if normal_only and "anomaly_label" in working_df.columns:
        working_df = working_df[
            working_df["anomaly_label"].isna()
            | (working_df["anomaly_label"].astype(str).str.lower() == "normal")
        ]

    sequences = []

    for group_value, group_df in working_df.groupby(group_by):
        group_df = group_df.sort_values("ts")
        templates = group_df["template"].tolist()
        timestamps = group_df["ts"].astype(str).tolist()

        if len(templates) < window_size:
            continue

        start = 0
        while start + window_size <= len(templates):
            end = start + window_size
            seq = templates[start:end]

            sequences.append(
                {
                    "group": str(group_value),
                    "start_ts": timestamps[start],
                    "end_ts": timestamps[end - 1],
                    "templates": seq,
                }
            )

            start += step_size

    return sequences


def split_sequences(
    sequences: list[dict],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    total = len(sequences)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = sequences[:train_end]
    val_data = sequences[train_end:val_end]
    test_data = sequences[val_end:]

    return train_data, val_data, test_data


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_type", choices=["csv", "sqlite"], required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--table", default="logs")
    parser.add_argument("--group_by", default="host", choices=["host", "user"])
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--step_size", type=int, default=1)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--normal_only", action="store_true")

    args = parser.parse_args()

    if args.input_type == "csv":
        df = load_from_csv(args.input_path)
    else:
        df = load_from_sqlite(args.input_path, table=args.table)

    validate_columns(df)
    df = normalise_dataframe(df)

    vocab = build_event_vocab(df)

    sequences = build_sequences(
        df=df,
        group_by=args.group_by,
        window_size=args.window_size,
        step_size=args.step_size,
        normal_only=args.normal_only,
    )

    encoded_sequences = []
    for item in sequences:
        encoded_sequences.append(
            {
                "group": item["group"],
                "start_ts": item["start_ts"],
                "end_ts": item["end_ts"],
                "templates": item["templates"],
                "event_ids": encode_sequence(item["templates"], vocab),
            }
        )

    train_data, val_data, test_data = split_sequences(
        encoded_sequences,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    output_dir = Path(args.output_dir)
    save_json(output_dir / "vocab.json", vocab)
    save_json(output_dir / "train.json", train_data)
    save_json(output_dir / "val.json", val_data)
    save_json(output_dir / "test.json", test_data)

    print(f"Total sequences: {len(encoded_sequences)}")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()