import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    # Sets up command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_repo", default=None)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # Gets the adapter and output folder paths.
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    # Creates the output folder if it does not exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loads the tokenizer for the base model.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Loads the base Qwen model.
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Loads the LoRA adapter onto the base model.
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
    )

    # Merges the LoRA adapter into the base model.
    print("Merging LoRA adapter into base model...")
    merged_model = model.merge_and_unload()

    # Saves the merged model and tokenizer locally.
    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="2GB",
    )
    tokenizer.save_pretrained(str(output_dir))

    # Pushes the merged model to Hugging Face if requested.
    if args.push_to_hub:
        if not args.hub_repo:
            raise ValueError("--hub_repo is required with --push_to_hub")

        print(f"Pushing merged model to Hugging Face: {args.hub_repo}")

        # Uploads the merged model to the selected Hugging Face repo.
        merged_model.push_to_hub(
            args.hub_repo,
            private=args.private,
            safe_serialization=True,
            max_shard_size="2GB",
        )

        # Uploads the tokenizer to the same Hugging Face repo.
        tokenizer.push_to_hub(args.hub_repo, private=args.private)

    print("Done.")


if __name__ == "__main__":
    # Runs the script.
    main()