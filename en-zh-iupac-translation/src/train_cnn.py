"""Train the CNN attention seq2seq model for chemical name translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Convolution1D,
    Dense,
    Dot,
    Input,
)
from tensorflow.keras.models import Model

from data_loader import DEFAULT_DATA_PATH, load_pairs, vectorize_pairs
from train_lstm import build_training_arrays


def build_cnn_model(
    num_encoder_tokens: int,
    num_decoder_tokens: int,
    latent_dim: int = 256,
    latent_dim_out: int = 64,
) -> Model:
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    x_encoder = Convolution1D(latent_dim, kernel_size=3, activation="relu", padding="causal")(encoder_inputs)
    x_encoder = Convolution1D(
        latent_dim, kernel_size=3, activation="relu", padding="causal", dilation_rate=2
    )(x_encoder)
    x_encoder = Convolution1D(
        latent_dim, kernel_size=3, activation="relu", padding="causal", dilation_rate=4
    )(x_encoder)

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    x_decoder = Convolution1D(latent_dim, kernel_size=3, activation="relu", padding="causal")(decoder_inputs)
    x_decoder = Convolution1D(
        latent_dim, kernel_size=3, activation="relu", padding="causal", dilation_rate=2
    )(x_decoder)
    x_decoder = Convolution1D(
        latent_dim, kernel_size=3, activation="relu", padding="causal", dilation_rate=4
    )(x_decoder)

    attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
    attention = Activation("softmax")(attention)
    context = Dot(axes=[2, 1])([attention, x_encoder])
    decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

    decoder_outputs = Convolution1D(
        latent_dim_out, kernel_size=3, activation="relu", padding="causal"
    )(decoder_combined_context)
    decoder_outputs = Convolution1D(
        latent_dim_out, kernel_size=3, activation="relu", padding="causal"
    )(decoder_outputs)
    decoder_outputs = Dense(num_decoder_tokens, activation="softmax")(decoder_outputs)

    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def train(
    direction: str = "en2ch",
    data_path: str | Path | None = None,
    output_dir: str | Path = "models/cnn_en2ch",
    epochs: int = 100,
    batch_size: int = 64,
    latent_dim: int = 256,
    validation_split: float = 0.2,
    max_samples: int | None = None,
) -> Path:
    pairs = load_pairs(direction=direction, data_path=data_path)
    if max_samples is not None:
        pairs = pairs[:max_samples]

    (
        input_texts,
        target_texts,
        input_token_index,
        target_token_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
    ) = vectorize_pairs(pairs)

    print(f"Direction: {direction}")
    print(f"Samples: {len(input_texts)}")
    print(f"Encoder tokens: {len(input_token_index)}")
    print(f"Decoder tokens: {len(target_token_index)}")

    encoder_input_data, decoder_input_data, decoder_target_data = build_training_arrays(
        input_texts,
        target_texts,
        input_token_index,
        target_token_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
    )

    model = build_cnn_model(len(input_token_index), len(target_token_index), latent_dim)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=2,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / "model.keras"
    model.save(model_file)

    metadata = {
        "model_type": "cnn",
        "direction": direction,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "latent_dim": latent_dim,
        "supports_inference": False,
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model to {model_file}")
    print("Note: CNN model is saved for training/evaluation only. Use LSTM for CLI translation.")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN chemical name translator")
    parser.add_argument("--direction", choices=["en2ch", "ch2en"], default="en2ch")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("models/cnn_en2ch"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    train(
        direction=args.direction,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
