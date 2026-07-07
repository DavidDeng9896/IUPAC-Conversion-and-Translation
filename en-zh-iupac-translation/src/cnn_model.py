# **References**
# - https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html
# - https://github.com/keras-team/keras/blob/master/examples/cnn_seq2seq.py
# Note:
# Scripts that were used to generate the models described in
# "Neural Machine Translation of Chemical Nomenclature between English and Chinese" Tingjun Xu et al.
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
import numpy as np
import pymssql
import matplotlib.pyplot as plt

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
latent_dim_out = 64  # Latent dimensionality of the output encoding space.

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
conn = pymssql.connect(server="DatabaseServer", database="TrainDataSet")
cursor = conn.cursor()
cursor.execute('SELECT SourceName, TargetName from TrainData')  # Read the train data set from sql server database.
row = cursor.fetchone()
while row:
    input_text = row[0]  # Source Names
    target_text = row[1]  # Target Names
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    input_text = input_text + ' '
    target_text = '\t' + target_text + ' ' + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
    row = cursor.fetchone()
conn.close()
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
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
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2, verbose=2)
# Save model
model.save('model name' + '.h5')
