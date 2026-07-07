"""Load EN↔ZH chemical name pairs from the paper supplementary Excel file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

Direction = Literal["en2ch", "ch2en"]

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "training_dataset.xlsx"

SHEET_BY_DIRECTION: dict[Direction, str] = {
    "en2ch": "Data_EN2CH",
    "ch2en": "Data_CH2EN",
}


def load_pairs(
    direction: Direction = "en2ch",
    data_path: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Return (source, target) name pairs for the given translation direction."""
    path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    sheet = SHEET_BY_DIRECTION[direction]
    df = pd.read_excel(path, sheet_name=sheet)
    if "Source Name" not in df.columns or "Target Name" not in df.columns:
        raise ValueError(f"Unexpected columns in {sheet}: {list(df.columns)}")

    pairs: list[tuple[str, str]] = []
    for source, target in zip(df["Source Name"], df["Target Name"]):
        source_text = str(source).strip()
        target_text = str(target).strip()
        if source_text and target_text and source_text.lower() != "nan" and target_text.lower() != "nan":
            pairs.append((source_text, target_text))
    return pairs


def vectorize_pairs(
    pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str], dict[str, int], dict[str, int], int, int]:
    """Convert pairs into seq2seq training tensors metadata."""
    input_texts: list[str] = []
    target_texts: list[str] = []
    input_characters: set[str] = set()
    target_characters: set[str] = set()

    for source, target in pairs:
        input_text = source + " "
        target_text = "\t" + target + " \n"
        input_texts.append(input_text)
        target_texts.append(target_text)
        input_characters.update(input_text)
        target_characters.update(target_text)

    input_characters_sorted = sorted(input_characters)
    target_characters_sorted = sorted(target_characters)
    input_token_index = {char: i for i, char in enumerate(input_characters_sorted)}
    target_token_index = {char: i for i, char in enumerate(target_characters_sorted)}
    max_encoder_seq_length = max(len(txt) for txt in input_texts)
    max_decoder_seq_length = max(len(txt) for txt in target_texts)

    return (
        input_texts,
        target_texts,
        input_token_index,
        target_token_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
    )
