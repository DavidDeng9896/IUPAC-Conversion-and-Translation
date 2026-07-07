"""Translate chemical names with a trained LSTM seq2seq model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model


def load_metadata(model_dir: Path) -> dict:
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def encode_input(
    text: str,
    input_token_index: dict[str, int],
    max_encoder_seq_length: int,
) -> np.ndarray:
    input_text = text + " "
    encoder_input = np.zeros((1, max_encoder_seq_length, len(input_token_index)), dtype="float32")
    truncated = input_text[-max_encoder_seq_length:]
    for t, char in enumerate(truncated):
        if char in input_token_index:
            encoder_input[0, t, input_token_index[char]] = 1.0
    return encoder_input


def decode_sequence(
    input_seq: np.ndarray,
    encoder_model,
    decoder_model,
    target_token_index: dict[str, int],
    reverse_target_token_index: dict[int, str],
    max_decoder_seq_length: int,
) -> str:
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, len(target_token_index)), dtype="float32")
    if "\t" not in target_token_index:
        raise ValueError("Target vocabulary is missing start token '\\t'")
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    decoded_chars: list[str] = []

    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))
        sampled_char = reverse_target_token_index.get(sampled_token_index, "")

        if sampled_char == "\n":
            break
        if sampled_char and sampled_char not in {"\t", " "}:
            decoded_chars.append(sampled_char)

        target_seq = np.zeros((1, 1, len(target_token_index)), dtype="float32")
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return "".join(decoded_chars)


def translate(model_dir: Path, text: str) -> str:
    metadata = load_metadata(model_dir)
    encoder_model = load_model(model_dir / "encoder.keras")
    decoder_model = load_model(model_dir / "decoder.keras")

    reverse_target_token_index = {i: c for c, i in metadata["target_token_index"].items()}
    input_seq = encode_input(
        text,
        metadata["input_token_index"],
        metadata["max_encoder_seq_length"],
    )
    return decode_sequence(
        input_seq,
        encoder_model,
        decoder_model,
        metadata["target_token_index"],
        reverse_target_token_index,
        metadata["max_decoder_seq_length"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate chemical names with a trained LSTM model")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with encoder.keras, decoder.keras, metadata.json")
    parser.add_argument("--text", type=str, help="Single name to translate")
    parser.add_argument("--input-file", type=Path, help="Text file with one name per line")
    args = parser.parse_args()

    if not args.text and not args.input_file:
        parser.error("Provide --text or --input-file")

    if args.text:
        print(translate(args.model_dir, args.text))
        return

    for line in args.input_file.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name:
            print(f"{name}\t{translate(args.model_dir, name)}")


if __name__ == "__main__":
    main()
