"""Download and checksum the published LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


EXPECTED_SHA256 = "e00d5e137ed29a8487c963c534b52ce9196faa489517fb00e24fd2f3a5554513"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("WEIGHTS_URL"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/final/adapter_model.safetensors"))
    args = parser.parse_args()
    if not args.url:
        raise SystemExit("Pass --url or set WEIGHTS_URL (see README).")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".download")
    try:
        with urllib.request.urlopen(args.url) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        actual = sha256(temporary)
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"weight checksum mismatch: {actual}")
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {args.output}: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()

