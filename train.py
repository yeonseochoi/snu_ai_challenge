"""Reproducible CLI version of the training cells in final_code.ipynb."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from common import USER_TEXT_TMPL, apply_perm, load_split, sample_perms


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--model",
        default=None,
        help="Base model ID or local directory. Use a local directory for offline training.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def swift_content(sentence: str) -> str:
    prefix = "".join(f"<image>\nImage {i}\n" for i in range(1, 5))
    return prefix + USER_TEXT_TMPL.format(sentence=sentence)


def write_jsonl(samples, path: Path, k_aug: int, seed: int):
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as output:
        for sample in samples:
            for sigma in sample_perms(k_aug, rng):
                frames, target = apply_perm(sample, sigma)
                record = {
                    "messages": [
                        {"role": "user", "content": swift_content(sample["caption"])},
                        {"role": "assistant", "content": str(target)},
                    ],
                    "images": frames,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["data"]["seed"])
    all_train = load_split(args.data_dir.resolve(), "train")
    indices = (
        pd.Series(range(len(all_train)))
        .sample(frac=1.0, random_state=seed)
        .tolist()
    )
    val_count = int(len(all_train) * float(config["data"]["val_ratio"]))
    val = [all_train[index] for index in indices[:val_count]]
    train = [all_train[index] for index in indices[val_count:]]
    if args.smoke_test:
        train, val = train[:16], val[:4]

    work_dir = args.output_dir.resolve() / "prepared"
    work_dir.mkdir(parents=True, exist_ok=True)
    train_jsonl, val_jsonl = work_dir / "train_aug.jsonl", work_dir / "val.jsonl"
    write_jsonl(train, train_jsonl, int(config["training"]["k_aug"]), seed)
    write_jsonl(val, val_jsonl, 1, seed)
    print(f"Prepared train={len(train)}, val={len(val)} in {work_dir}")
    if args.prepare_only:
        return

    model, training = config["model"], config["training"]
    model_source = args.model or model["id"]
    run_dir = args.output_dir.resolve() / training["run_name"]
    environment = os.environ.copy()
    environment.update(
        {
            "MIN_PIXELS": str(model["min_pixels"]),
            "MAX_PIXELS": str(model["max_pixels"]),
            "IMAGE_MIN_TOKEN_NUM": "128",
            "IMAGE_MAX_TOKEN_NUM": "320",
        }
    )
    command = [
        "swift", "sft",
        "--model", model_source,
        "--dataset", str(train_jsonl),
        "--val_dataset", str(val_jsonl),
        "--tuner_type", "lora",
        "--lora_rank", str(training["lora_rank"]),
        "--lora_alpha", str(training["lora_alpha"]),
        "--lora_dropout", str(training["lora_dropout"]),
        "--target_modules", "all-linear",
        "--freeze_vit", "false",
        "--freeze_aligner", "false",
        "--learning_rate", str(training["learning_rate"]),
        "--vit_lr", str(training["vit_learning_rate"]),
        "--num_train_epochs", str(training["epochs"]),
        "--per_device_train_batch_size", str(training["batch_size"]),
        "--gradient_accumulation_steps", str(training["gradient_accumulation_steps"]),
        "--packing", "false",
        "--torch_dtype", "bfloat16",
        "--attn_impl", "sdpa",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "true",
        "--save_steps", str(training["save_steps"]),
        "--save_total_limit", "3",
        "--logging_steps", "10",
        "--max_length", "4096",
        "--output_dir", str(run_dir),
        "--seed", str(seed),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
