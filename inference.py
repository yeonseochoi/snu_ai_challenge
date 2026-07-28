"""Deterministic TTA inference for the final Qwen3-VL LoRA model."""

from __future__ import annotations

import argparse
import ast
import random
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from peft import PeftModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from common import (
    ALL_PERMS,
    apply_perm,
    build_messages,
    inverse_perm,
    load_split,
    sample_perms,
    stable_sample_seed,
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, default=Path("checkpoints/final"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--model",
        default=None,
        help="Base model ID or local directory. Use a local directory for offline evaluation.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/submission_tta2.csv"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def validate(frame: pd.DataFrame, expected_rows: int):
    if len(frame) != expected_rows or frame["Id"].astype(str).nunique() != expected_rows:
        raise ValueError("submission row count or Id uniqueness check failed")
    for value in frame["Answer"]:
        if sorted(ast.literal_eval(str(value))) != [1, 2, 3, 4]:
            raise ValueError(f"invalid answer: {value}")


def main():
    args = arguments()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_config, infer_config = config["model"], config["inference"]
    model_source = args.model or model_config["id"]
    chunk = int(infer_config["candidate_chunk_size"])
    if 24 % chunk:
        raise ValueError("candidate_chunk_size must divide 24")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    test_set = load_split(args.data_dir.resolve(), "test")
    adapter_dir = args.adapter_dir.resolve()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_dir / filename).is_file():
            raise FileNotFoundError(adapter_dir / filename)

    processor = AutoProcessor.from_pretrained(
        model_source,
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_source,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).eval()
    candidate_ids = [
        processor.tokenizer.encode(str(p) + "<|im_end|>", add_special_tokens=False)
        for p in ALL_PERMS
    ]
    if len({len(ids) for ids in candidate_ids}) != 1:
        raise ValueError("candidate token lengths differ; check the tokenizer revision")
    candidates_cpu = torch.tensor(candidate_ids)

    @torch.no_grad()
    def score_view(sentence, paths):
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        inputs = processor.apply_chat_template(
            build_messages(sentence, images),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        scores = torch.empty(24)
        for start in range(0, 24, chunk):
            candidates = candidates_cpu[start : start + chunk].to("cuda")
            prefix = model(**inputs, use_cache=True)
            cache = prefix.past_key_values
            cache.batch_repeat_interleave(len(candidates))
            continuation = model(
                input_ids=candidates, past_key_values=cache, use_cache=False
            )
            logits = torch.cat(
                [
                    prefix.logits[:, -1:, :].expand(len(candidates), -1, -1),
                    continuation.logits[:, :-1, :],
                ],
                dim=1,
            )
            logp = torch.log_softmax(logits.float(), dim=-1)
            scores[start : start + chunk] = (
                logp.gather(-1, candidates.unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
                .cpu()
            )
            del prefix, continuation, cache, logits, logp
        return scores

    @torch.no_grad()
    def predict(sample):
        rng = random.Random(
            stable_sample_seed(sample["id"], int(infer_config["seed"]))
        )
        aggregate = {}
        for sigma in sample_perms(int(infer_config["k_tta"]), rng):
            paths, _ = apply_perm(sample, sigma)
            scores = score_view(sample["caption"], paths)
            for index, candidate in enumerate(ALL_PERMS):
                original = tuple(sigma[position - 1] for position in candidate)
                aggregate[original] = aggregate.get(original, 0.0) + scores[index].item()
        return list(max(aggregate, key=aggregate.get))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.stem + "_partial.csv")
    if args.resume and partial.exists():
        rows = pd.read_csv(partial, dtype={"Id": str}).to_dict("records")
    else:
        rows = []
    completed = {str(row["Id"]) for row in rows}
    start_time = time.perf_counter()
    for sample in tqdm(test_set, desc="test TTA=2"):
        if sample["id"] in completed:
            continue
        rows.append(
            {"Id": sample["id"], "Answer": str(inverse_perm(predict(sample)))}
        )
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(partial, index=False)

    order = {sample["id"]: index for index, sample in enumerate(test_set)}
    result = pd.DataFrame(rows)
    result["_order"] = result["Id"].astype(str).map(order)
    result = result.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    validate(result, len(test_set))
    result.to_csv(partial, index=False)
    result.to_csv(args.output, index=False)
    print(f"Saved {args.output} in {(time.perf_counter() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
