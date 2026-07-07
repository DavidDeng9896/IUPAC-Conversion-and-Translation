"""Load local CSV training data for seq2seq models."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from keras.utils import Sequence

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_translation_pairs(csv_path: str | Path) -> tuple[list[str], list[str]]:
    input_texts: list[str] = []
    target_texts: list[str] = []

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source = (row.get("SourceName") or "").strip()
            target = (row.get("TargetName") or "").strip()
            if not source or not target:
                continue

            input_text = source + " "
            target_text = "\t" + target + " " + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)

    return input_texts, target_texts


def default_dataset_path(direction: str, split: str = "train") -> Path:
    if direction not in {"ch2en", "en2ch"}:
        raise ValueError("direction must be 'ch2en' or 'en2ch'")
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'")
    return DEFAULT_DATA_DIR / f"{direction}_{split}.csv"


def build_token_indices(
    input_texts: list[str], target_texts: list[str]
) -> tuple[list[str], list[str], dict[str, int], dict[str, int], int, int, int, int]:
    input_characters = sorted({char for text in input_texts for char in text})
    target_characters = sorted({char for text in target_texts for char in text})
    input_token_index = {char: index for index, char in enumerate(input_characters)}
    target_token_index = {char: index for index, char in enumerate(target_characters)}
    max_encoder_seq_length = max(len(text) for text in input_texts)
    max_decoder_seq_length = max(len(text) for text in target_texts)
    return (
        input_characters,
        target_characters,
        input_token_index,
        target_token_index,
        len(input_characters),
        len(target_characters),
        max_encoder_seq_length,
        max_decoder_seq_length,
    )


def vectorize_batch(
    input_texts: list[str],
    target_texts: list[str],
    input_token_index: dict[str, int],
    target_token_index: dict[str, int],
    num_encoder_tokens: int,
    num_decoder_tokens: int,
    max_encoder_seq_length: int | None = None,
    max_decoder_seq_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_encoder_seq_length is None:
        max_encoder_seq_length = max(len(text) for text in input_texts)
    if max_decoder_seq_length is None:
        max_decoder_seq_length = max(len(text) for text in target_texts)
    batch_size = len(input_texts)

    encoder_input_data = np.zeros(
        (batch_size, max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (batch_size, max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (batch_size, max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for timestep, char in enumerate(input_text):
            encoder_input_data[index, timestep, input_token_index[char]] = 1.0
        encoder_input_data[index, timestep + 1 :, input_token_index[" "]] = 1.0

        for timestep, char in enumerate(target_text):
            decoder_input_data[index, timestep, target_token_index[char]] = 1.0
            if timestep > 0:
                decoder_target_data[index, timestep - 1, target_token_index[char]] = 1.0
        decoder_input_data[index, timestep + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[index, timestep:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data


def vectorize_texts(
    input_texts: list[str],
    target_texts: list[str],
    input_token_index: dict[str, int],
    target_token_index: dict[str, int],
    num_encoder_tokens: int,
    num_decoder_tokens: int,
    max_encoder_seq_length: int,
    max_decoder_seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype="float32",
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype="float32",
    )

    for index, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for timestep, char in enumerate(input_text):
            encoder_input_data[index, timestep, input_token_index[char]] = 1.0
        encoder_input_data[index, timestep + 1 :, input_token_index[" "]] = 1.0

        for timestep, char in enumerate(target_text):
            decoder_input_data[index, timestep, target_token_index[char]] = 1.0
            if timestep > 0:
                decoder_target_data[index, timestep - 1, target_token_index[char]] = 1.0
        decoder_input_data[index, timestep + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[index, timestep:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data


class TranslationBatchGenerator(Sequence):
    """Yield one-hot batches with per-batch dynamic padding."""

    def __init__(
        self,
        input_texts: list[str],
        target_texts: list[str],
        input_token_index: dict[str, int],
        target_token_index: dict[str, int],
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        batch_size: int,
        max_encoder_seq_length: int,
        max_decoder_seq_length: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if len(input_texts) != len(target_texts):
            raise ValueError("input_texts and target_texts must have the same length")

        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.batch_size = batch_size
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length
        self.shuffle = shuffle
        self.seed = seed
        self.indices = np.arange(len(input_texts))
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.input_texts) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, index: int) -> tuple[list[np.ndarray], np.ndarray]:
        start = index * self.batch_size
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        batch_input_texts = [self.input_texts[i] for i in batch_indices]
        batch_target_texts = [self.target_texts[i] for i in batch_indices]
        encoder_input_data, decoder_input_data, decoder_target_data = vectorize_batch(
            batch_input_texts,
            batch_target_texts,
            self.input_token_index,
            self.target_token_index,
            self.num_encoder_tokens,
            self.num_decoder_tokens,
            self.max_encoder_seq_length,
            self.max_decoder_seq_length,
        )
        return (encoder_input_data, decoder_input_data), decoder_target_data
