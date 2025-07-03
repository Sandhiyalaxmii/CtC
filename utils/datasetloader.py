import tensorflow as tf
import tensorflow_datasets as tfds

IMG_HEIGHT, IMG_WIDTH = 299, 299
BUFFER_SIZE = 1000

def get_image_label(example):
    caption = example["captions"]["text"][0]
    img = example["image"]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255
    return {"image_tensor": img, "caption": caption}

def add_start_end_token(data):
    start = tf.convert_to_tensor("<start>")
    end = tf.convert_to_tensor("<end>")
    data["caption"] = tf.strings.join([start, data["caption"], end], separator=" ")
    return data

def load_dataset():
    trainds = tfds.load("coco_captions", split="train")
    trainds = trainds.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE)
    trainds = trainds.map(add_start_end_token)
    trainds = trainds.shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    return trainds
