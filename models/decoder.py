import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, GRU, Attention, Add, LayerNormalization, Dense
from tensorflow.keras.models import Model

def build_decoder(VOCAB_SIZE, MAX_CAPTION_LEN, ATTENTION_DIM):
    word_input = Input(shape=(MAX_CAPTION_LEN,), name="words")
    encoder_output_input = Input(shape=(64, ATTENTION_DIM), name="encoder_output_input")

    embed_x = Embedding(VOCAB_SIZE, ATTENTION_DIM)(word_input)
    decoder_gru = GRU(ATTENTION_DIM, return_sequences=True, return_state=True)
    gru_output, gru_state = decoder_gru(embed_x)

    decoder_attention = Attention()
    context_vector = decoder_attention([gru_output, encoder_output_input])
    addition = Add()([gru_output, context_vector])
    layer_norm_out = LayerNormalization(axis=-1)(addition)
    decoder_output = Dense(VOCAB_SIZE)(layer_norm_out)

    return Model(inputs=[word_input, encoder_output_input], outputs=decoder_output)
