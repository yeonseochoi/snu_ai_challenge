"""Validate local final artifacts against the recorded submission."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

import pandas as pd


EXPECTED = {
    "adapter_model.safetensors": "e00d5e137ed29a8487c963c534b52ce9196faa489517fb00e24fd2f3a5554513",
    "adapter_config.json": "622be38bc0afa35f9fdb5224b450c193b121d4635f4232afb0b8b3f1e56b4af5",
    "submission_tta2.csv": "6b9d4bb2a90bcc6a162bf58d47888e1c202315607a5179c9b653a03459a08ab8",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, default=Path("checkpoint-2265"))
    parser.add_argument("--submission", type=Path, default=Path("checkpoint-2265/submission_tta2.csv"))
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Fail instead of skipping when adapter_model.safetensors is absent.",
    )
    args = parser.parse_args()
    paths = {
        "adapter_model.safetensors": args.adapter_dir / "adapter_model.safetensors",
        "adapter_config.json": args.adapter_dir / "adapter_config.json",
        "submission_tta2.csv": args.submission,
    }
    for name, path in paths.items():
        if not path.exists() and name == "adapter_model.safetensors":
            if args.require_weights:
                raise SystemExit(f"FAIL missing required weight file: {path}")
            print(f"SKIP {path}: download the weight file first")
            continue
        actual = digest(path)
        if actual != EXPECTED[name]:
            raise SystemExit(f"FAIL {path}: {actual} != {EXPECTED[name]}")
        print(f"OK   {path}: {actual}")
    frame = pd.read_csv(args.submission, dtype={"Id": str})
    if len(frame) != 819 or frame["Id"].nunique() != 819:
        raise SystemExit("FAIL submission must contain 819 unique Id values")
    if any(sorted(ast.literal_eval(value)) != [1, 2, 3, 4] for value in frame["Answer"]):
        raise SystemExit("FAIL invalid Answer permutation")
    print("OK   submission schema: 819 unique rows and valid permutations")
    print("Recorded private test leaderboard score: 0.89528")


if __name__ == "__main__":
    main()
