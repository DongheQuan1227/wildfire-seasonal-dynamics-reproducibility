#!/usr/bin/env python3
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "00_Input_data"
MANIFEST = INPUT_DIR / "data_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"Missing manifest: {MANIFEST}")
        return 1
    failures = []
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            path = INPUT_DIR / row["file"]
            if not path.is_file():
                failures.append(f"missing: {row['file']}")
                continue
            size = path.stat().st_size
            if size != int(row["size_bytes"]):
                failures.append(f"size mismatch: {row['file']} ({size} != {row['size_bytes']})")
                continue
            observed = sha256(path)
            if observed != row["sha256"]:
                failures.append(f"SHA256 mismatch: {row['file']}")
    if failures:
        print("Input-manifest verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Input-data manifest verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
