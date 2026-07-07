"""Train the LSTM seq2seq model for chemical name translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model, load_model

from data_loader import DEFAULT_DATA_PATH, load_pairs, vectorize_pairs


def build_training_arrays(
    input_texts: list[str],
    target_texts: list[str],
    input_token_index: dict[str, int],
    target_token_index: dict[str, int],
    max_encoder_seq_length: int,
    max_decoder_seq_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_encoder_tokens = len(input_token_index)
    num_decoder_tokens = len(target_token_index)

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

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
        encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0

        for t, char in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[char]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
        decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
        decoder_target_data[i, t:, target_token_index[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data


def build_lstm_model(
    num_encoder_tokens: int,
    num_decoder_tokens: int,
    latent_dim: int = 256,
) -> Model:
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    _, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def build_inference_models_from_training_model(
    model: Model,
    latent_dim: int,
    num_decoder_tokens: int,
) -> tuple[Model, Model]:
    encoder_inputs = model.input[0]
    _, state_h_enc, state_c_enc = model.get_layer(index=2).output
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.get_layer(index=3)
    decoder_dense = model.get_layer(index=4)
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs, state_h, state_c],
    )
    return encoder_model, decoder_model


def save_metadata(
    output_dir: Path,
    direction: str,
    input_token_index: dict[str, int],
    target_token_index: dict[str, int],
    max_encoder_seq_length: int,
    max_decoder_seq_length: int,
    latent_dim: int,
) -> None:
    metadata = {
        "model_type": "lstm",
        "direction": direction,
        "input_token_index": input_token_index,
        "target_token_index": target_token_index,
        "max_encoder_seq_length": max_encoder_seq_length,
        "max_decoder_seq_length": max_decoder_seq_length,
        "latent_dim": latent_dim,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(
    direction: str = "en2ch",
    data_path: str | Path | None = None,
    output_dir: str | Path = "models/lstm_en2ch",
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
    print(f"Max encoder length: {max_encoder_seq_length}")
    print(f"Max decoder length: {max_decoder_seq_length}")

    encoder_input_data, decoder_input_data, decoder_target_data = build_training_arrays(
        input_texts,
        target_texts,
        input_token_index,
        target_token_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
    )

    model = build_lstm_model(len(input_token_index), len(target_token_index), latent_dim)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
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

    encoder_model, decoder_model = build_inference_models_from_training_model(
        model,
        latent_dim,
        len(target_token_index),
    )
    encoder_model.save(output_path / "encoder.keras")
    decoder_model.save(output_path / "decoder.keras")

    save_metadata(
        output_path,
        direction,
        input_token_index,
        target_token_index,
        max_encoder_seq_length,
        max_decoder_seq_length,
        latent_dim,
    )
    print(f"Saved model to {model_file}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM chemical name translator")
    parser.add_argument("--direction", choices=["en2ch", "ch2en"], default="en2ch")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("models/lstm_en2ch"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=None, help="Use a subset for quick tests")
    args = parser.parse_args()

    train(
        direction=args.direction,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        validation_split=args.validation_split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
