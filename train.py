import tensorflow as tf
from models.encoder import build_encoder
from models.decoder import build_decoder
from models.feature_extractor import get_feature_extractor
from utils.datasetloader import load_dataset
from utils.tokenizer import build_tokenizer, get_lookup_tables
from utils.loss import loss_function

# Hyperparameters
VOCAB_SIZE = 20000
ATTENTION_DIM = 512
MAX_CAPTION_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10

# Load dataset
train_ds = load_dataset()

# Tokenizer
tokenizer = build_tokenizer(train_ds)
word_to_index, index_to_word = get_lookup_tables(tokenizer)

# Prepare batched dataset
def create_ds_fn(data):
    img_tensor = data["image_tensor"]
    caption = tokenizer(data["caption"])
    target = tf.roll(caption, -1, 0)
    zeros = tf.zeros([1], dtype=tf.int64)
    target = tf.concat((target[:-1], zeros), axis=-1)
    return (img_tensor, caption), target

batched_ds = (
    train_ds.map(create_ds_fn)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Build models
encoder = build_encoder()
decoder = build_decoder(VOCAB_SIZE, MAX_CAPTION_LEN, ATTENTION_DIM)

# Decoder prediction model for step-by-step prediction
gru = tf.keras.layers.GRU(ATTENTION_DIM, return_sequences=True, return_state=True)
gru_state_input = tf.keras.Input(shape=(ATTENTION_DIM,), name="gru_state_input")
word_input = tf.keras.Input(shape=(MAX_CAPTION_LEN,), name="word_input")
encoder_output_input = tf.keras.Input(shape=(64, ATTENTION_DIM), name="encoder_output_input")

embed_x = decoder.layers[1](word_input)
gru_output, gru_state = gru(embed_x, initial_state=gru_state_input)

attention_layer = tf.keras.layers.Attention()
context_vector = attention_layer([gru_output, encoder_output_input])
added = tf.keras.layers.Add()([gru_output, context_vector])
layer_norm = tf.keras.layers.LayerNormalization(axis=-1)(added)
output = tf.keras.layers.Dense(VOCAB_SIZE)(layer_norm)

decoder_pred_model = tf.keras.Model(
    inputs=[word_input, gru_state_input, encoder_output_input],
    outputs=[output, gru_state]
)

# Training setup
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(img_tensor, target):
    loss = 0
    with tf.GradientTape() as tape:
        gru_state = tf.zeros((BATCH_SIZE, ATTENTION_DIM))
        dec_input = tf.expand_dims([word_to_index("<start>")] * BATCH_SIZE, 1)
        features = encoder(img_tensor, training=False)
        for i in range(1, target.shape[1]):
            predictions, gru_state = decoder_pred_model([dec_input, gru_state, features], training=True)
            loss += loss_function(target[:, i], predictions[:, 0, :])
            dec_input = tf.expand_dims(target[:, i], 1)
    batch_loss = loss / int(target.shape[1])
    trainable_vars = decoder_pred_model.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return batch_loss

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for (batch, (img_tensor, caption)), target in batched_ds.enumerate():
        batch_loss = train_step(img_tensor, target)
        total_loss += batch_loss
    print(f"Epoch {epoch + 1} Loss {total_loss.numpy() / (batch + 1):.4f}")

# Save model weights
encoder.save_weights('encoder_weights')
decoder_pred_model.save_weights('decoder_pred_weights')

print("Training completed and model saved.")
