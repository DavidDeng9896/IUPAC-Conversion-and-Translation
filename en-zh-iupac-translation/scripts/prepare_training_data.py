#!/usr/bin/env python3
"""Prepare training CSV files for EN-ZH IUPAC translation models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def has_cjk(text: str) -> bool:
    return bool(CJK_PATTERN.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_PATTERN.search(text))


def is_valid_ch2en_pair(source: str, target: str) -> bool:
    if not source or not target:
        return False
    if not has_cjk(source):
        return False
    if not has_latin(target):
        return False
    if has_cjk(target):
        return False
    return True


def read_raw_pairs(input_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        source_key = "SourceName" if "SourceName" in reader.fieldnames else "Source Name"
        target_key = "TargetName" if "TargetName" in reader.fieldnames else "Target Name"
        for row in reader:
            source = (row.get(source_key) or "").strip()
            target = (row.get(target_key) or "").strip()
            pairs.append((source, target))
    return pairs


def clean_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    valid_pairs = [pair for pair in pairs if is_valid_ch2en_pair(*pair)]
    return sorted(set(valid_pairs))


def split_pairs(
    pairs: list[tuple[str, str]], val_ratio: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    if val_size == 0 and len(shuffled) > 1:
        val_size = 1
    return shuffled[val_size:], shuffled[:val_size]


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["SourceName", "TargetName"])
        writer.writeheader()
        for source, target in pairs:
            writer.writerow({"SourceName": source, "TargetName": target})


def reverse_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(target, source) for source, target in pairs]


def build_stats(
    raw_count: int,
    cleaned_count: int,
    train_count: int,
    val_count: int,
    direction: str,
) -> dict[str, int | str]:
    return {
        "direction": direction,
        "raw_rows": raw_count,
        "cleaned_unique_rows": cleaned_count,
        "train_rows": train_count,
        "val_rows": val_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "data" / "training_dataset.csv",
        help="Raw training dataset CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "data" / "processed",
        help="Directory for processed CSV files.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_pairs = read_raw_pairs(args.input)
    cleaned_pairs = clean_pairs(raw_pairs)
    train_pairs, val_pairs = split_pairs(cleaned_pairs, args.val_ratio, args.seed)

    ch2en_train = train_pairs
    ch2en_val = val_pairs
    en2ch_train = reverse_pairs(train_pairs)
    en2ch_val = reverse_pairs(val_pairs)

    output_dir = args.output_dir
    write_pairs(output_dir / "ch2en_train.csv", ch2en_train)
    write_pairs(output_dir / "ch2en_val.csv", ch2en_val)
    write_pairs(output_dir / "en2ch_train.csv", en2ch_train)
    write_pairs(output_dir / "en2ch_val.csv", en2ch_val)

    stats = {
        "input": str(args.input),
        "output_dir": str(output_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "datasets": [
            build_stats(len(raw_pairs), len(cleaned_pairs), len(ch2en_train), len(ch2en_val), "ch2en"),
            build_stats(len(raw_pairs), len(cleaned_pairs), len(en2ch_train), len(en2ch_val), "en2ch"),
        ],
    }
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
