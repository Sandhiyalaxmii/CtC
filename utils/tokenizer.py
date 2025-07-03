import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, StringLookup

VOCAB_SIZE = 20000
MAX_CAPTION_LEN = 64

def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs, r"[!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~]?", "")

def build_tokenizer(train_ds):
    tokenizer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        standardize=standardize,
        output_sequence_length=MAX_CAPTION_LEN
    )
    tokenizer.adapt(train_ds.map(lambda x: x["caption"]))
    return tokenizer

def get_lookup_tables(tokenizer):
    word_to_index = StringLookup(vocabulary=tokenizer.get_vocabulary())
    index_to_word = StringLookup(vocabulary=tokenizer.get_vocabulary(), invert=True)
    return word_to_index, index_to_word
