import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 299, 299, 3
FEATURES_SHAPE = (8, 8, 1536)
ATTENTION_DIM = 512

def build_encoder():
    feature_extractor = tf.keras.applications.InceptionResNetV2(include_top=False, weights="imagenet")
    feature_extractor.trainable = False

    image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    features = feature_extractor(image_input)
    x = Reshape((FEATURES_SHAPE[0] * FEATURES_SHAPE[1], FEATURES_SHAPE[2]))(features)
    encoder_output = Dense(ATTENTION_DIM, activation="relu")(x)

    return Model(inputs=image_input, outputs=encoder_output)
