# **References**
# - https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
# - https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
# Note:
# Scripts that were used to generate the models described in
# "Neural Machine Translation of Chemical Nomenclature between English and Chinese" Tingjun Xu et al.
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
from keras import metrics

from data_loader import (
    build_token_indices,
    default_dataset_path,
    load_translation_pairs,
    vectorize_texts,
)

batch_size = 64  # Batch size for training.3774
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
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

encoder_input_data, decoder_input_data, decoder_target_data = vectorize_texts(
    train_input_texts,
    train_target_texts,
    input_token_index,
    target_token_index,
    num_encoder_tokens,
    num_decoder_tokens,
    max_encoder_seq_length,
    max_decoder_seq_length,
)
val_encoder_input_data, val_decoder_input_data, val_decoder_target_data = vectorize_texts(
    val_input_texts,
    val_target_texts,
    input_token_index,
    target_token_index,
    num_encoder_tokens,
    num_decoder_tokens,
    max_encoder_seq_length,
    max_decoder_seq_length,
)
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([val_encoder_input_data, val_decoder_input_data],
                               val_decoder_target_data),
              verbose=2)
# Save model
model.save('model name' + '.h5')
