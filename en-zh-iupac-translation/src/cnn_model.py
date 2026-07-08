# **References**
# - https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
# - https://github.com/keras-team/keras/blob/master/examples/cnn_seq2seq.py
# Note:
# Scripts that were used to generate the models described in
# "Neural Machine Translation of Chemical Nomenclature between English and Chinese" Tingjun Xu et al.
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
import matplotlib.pyplot as plt

from data_loader import (
    TranslationBatchGenerator,
    build_token_indices,
    default_dataset_path,
    load_translation_pairs,
)

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
latent_dim_out = 64  # Latent dimensionality of the output encoding space.
direction = 'ch2en'  # ch2en or en2ch

train_input_texts, train_target_texts = load_translation_pairs(
    default_dataset_path(direction, 'train'))
val_input_texts, val_target_texts = load_translation_pairs(
    default_dataset_path(direction, 'val'))

(
    input_characters,
    target_characters,
    input_token_index,
    target_token_index,
    num_encoder_tokens,
    num_decoder_tokens,
    max_encoder_seq_length,
    max_decoder_seq_length,
) = build_token_indices(
    train_input_texts + val_input_texts,
    train_target_texts + val_target_texts,
)

print('Number of training samples:', len(train_input_texts))
print('Number of validation samples:', len(val_input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

train_generator = TranslationBatchGenerator(
    train_input_texts,
    train_target_texts,
    input_token_index,
    target_token_index,
    num_encoder_tokens,
    num_decoder_tokens,
    batch_size,
    max_encoder_seq_length,
    max_decoder_seq_length,
    shuffle=True,
)
val_generator = TranslationBatchGenerator(
    val_input_texts,
    val_target_texts,
    input_token_index,
    target_token_index,
    num_encoder_tokens,
    num_decoder_tokens,
    batch_size,
    max_encoder_seq_length,
    max_decoder_seq_length,
    shuffle=False,
)
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Encoder
x_encoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal')(encoder_inputs)
x_encoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_encoder)
x_encoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_encoder)

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Decoder
x_decoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal')(decoder_inputs)
x_decoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_decoder)
x_decoder = Convolution1D(latent_dim, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_decoder)
# Attention
attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
attention = Activation('softmax')(attention)

context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

decoder_outputs = Convolution1D(latent_dim_out, kernel_size=3, activation='relu',
                                padding='causal')(decoder_combined_context)
decoder_outputs = Convolution1D(latent_dim_out, kernel_size=3, activation='relu',
                                padding='causal')(decoder_outputs)
# Output
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    verbose=2,
)
# Save model
model.save('model name' + '.h5')
