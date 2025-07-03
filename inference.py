import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from models.encoder import build_encoder
from models.decoder import build_decoder
from utils.tokenizer import build_tokenizer, get_lookup_tables
from utils.beam_search import beam_search_predict
from utils.datasetloader import load_dataset

# Hyperparameters
VOCAB_SIZE = 20000
ATTENTION_DIM = 512
MAX_CAPTION_LEN = 64

# Load tokenizer
train_ds = load_dataset()
tokenizer = build_tokenizer(train_ds)
word_to_index, index_to_word = get_lookup_tables(tokenizer)

# Load models
encoder = build_encoder()
encoder.load_weights('encoder_weights')

decoder = build_decoder(VOCAB_SIZE, MAX_CAPTION_LEN, ATTENTION_DIM)
gru_state_input = tf.keras.Input(shape=(ATTENTION_DIM,), name="gru_state_input")
word_input = tf.keras.Input(shape=(MAX_CAPTION_LEN,), name="word_input")
encoder_output_input = tf.keras.Input(shape=(64, ATTENTION_DIM), name="encoder_output_input")

embed_x = decoder.layers[1](word_input)
gru_output, gru_state = decoder.layers[2](embed_x, initial_state=gru_state_input)

attention_layer = tf.keras.layers.Attention()
context_vector = attention_layer([gru_output, encoder_output_input])
added = tf.keras.layers.Add()([gru_output, context_vector])
layer_norm = tf.keras.layers.LayerNormalization(axis=-1)(added)
output = tf.keras.layers.Dense(VOCAB_SIZE)(layer_norm)

decoder_pred_model = tf.keras.Model(
    inputs=[word_input, gru_state_input, encoder_output_input],
    outputs=[output, gru_state]
)
decoder_pred_model.load_weights('decoder_pred_weights')

# Test image path
filename = "data/my_test_images/beach.png"  # Change this to your test image

img, caption = beam_search_predict(filename, encoder, decoder_pred_model, tokenizer, word_to_index)
print("Generated Caption:", " ".join(caption))

plt.imshow(img)
plt.axis('off')
plt.show()
