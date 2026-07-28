"""Shared, deterministic data and permutation utilities."""

from __future__ import annotations

import ast
import itertools
import random
import zlib
from pathlib import Path

import pandas as pd


ALL_PERMS = [list(p) for p in itertools.permutations([1, 2, 3, 4])]
USER_TEXT_TMPL = (
    'Story: "{sentence}"\n'
    "The 4 images above (Image 1 to Image 4) are shuffled frames from a single video. "
    "Reorder them into the correct chronological order so that they match the story. "
    "If the story states an explicit sequence (e.g. words like 'then', 'after', "
    "'finally', 'begins by'), use that order directly. "
    "If the story does NOT state an explicit sequence, it describes a single continuous "
    "action or scene: compare the images for visual progression cues such as body/object "
    "position, motion direction, pose changes, or accumulated change (e.g. water level, "
    "object displacement, distance covered), and order the frames from earliest to latest "
    "based on those visual cues. "
    "Answer ONLY with a Python list of the image numbers in chronological order. "
    "Example: [3, 1, 4, 2]"
)


def inverse_perm(permutation: list[int]) -> list[int]:
    inverse = [0] * len(permutation)
    for index, value in enumerate(permutation):
        inverse[value - 1] = index + 1
    return inverse


def stable_sample_seed(sample_id: object, seed: int = 0) -> int:
    return (zlib.crc32(str(sample_id).encode("utf-8")) + seed) & 0xFFFFFFFF


def sample_perms(k: int, rng: random.Random) -> list[list[int]]:
    if not 1 <= k <= len(ALL_PERMS):
        raise ValueError(f"k must be in [1, {len(ALL_PERMS)}]")
    identity = (1, 2, 3, 4)
    pool = [tuple(p) for p in ALL_PERMS if tuple(p) != identity]
    return [list(identity), *[list(p) for p in rng.sample(pool, k - 1)]]


def apply_perm(sample: dict, sigma: list[int]) -> tuple[list[str], list[int] | None]:
    frames = [sample["frames"][index - 1] for index in sigma]
    if sample.get("gt") is None:
        return frames, None
    inv = inverse_perm(sigma)
    return frames, [inv[index - 1] for index in sample["gt"]]


def load_split(data_dir: Path, split: str) -> list[dict]:
    csv_path = data_dir / f"{split}.csv"
    frame_dir = data_dir / split
    frame = pd.read_csv(csv_path)
    required = ["Id", "Sentence", *[f"Input_{i}" for i in range(1, 5)]]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    samples = []
    for _, row in frame.iterrows():
        sample_id = str(row["Id"])
        paths = [
            frame_dir / sample_id / str(row[f"Input_{i}"])
            for i in range(1, 5)
        ]
        missing_paths = [path for path in paths if not path.is_file()]
        if missing_paths:
            raise FileNotFoundError(missing_paths[0])
        gt = (
            inverse_perm(ast.literal_eval(str(row["Answer"])))
            if "Answer" in frame.columns
            else None
        )
        samples.append(
            {
                "id": sample_id,
                "caption": str(row["Sentence"]),
                "frames": [str(path) for path in paths],
                "gt": gt,
            }
        )
    return samples


def build_messages(sentence: str, images: list) -> list[dict]:
    content = []
    for index, image in enumerate(images, start=1):
        content.extend(
            [
                {"type": "image", "image": image},
                {"type": "text", "text": f"\nImage {index}\n"},
            ]
        )
    content.append({"type": "text", "text": USER_TEXT_TMPL.format(sentence=sentence)})
    return [{"role": "user", "content": content}]

