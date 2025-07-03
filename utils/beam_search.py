import tensorflow as tf
import numpy as np

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 299, 299, 3
MAX_CAPTION_LEN = 64

def beam_search_predict(filename, encoder, decoder_pred_model, tokenizer, word_to_index, beam_width=3):
    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=IMG_CHANNELS)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = img / 255
    features = encoder(tf.expand_dims(img, 0))
    start_token = word_to_index("<start>")
    end_token = word_to_index("<end>")
    sequences = [[list(), 0.0, tf.zeros((1, 512)), tf.expand_dims([start_token], 1)]]

    for _ in range(MAX_CAPTION_LEN):
        all_candidates = []
        for seq, score, gru_state, dec_input in sequences:
            if len(seq) > 0 and seq[-1] == end_token:
                all_candidates.append((seq, score, gru_state, dec_input))
                continue
            predictions, gru_state_new = decoder_pred_model([dec_input, gru_state, features])
            log_probs = tf.nn.log_softmax(predictions[0, 0]).numpy()
            top_k_ids = np.argsort(log_probs)[-beam_width:]
            for idx in top_k_ids:
                candidate = [seq + [idx], score - log_probs[idx], gru_state_new, tf.expand_dims([idx], 1)]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_width]

    best_seq = sequences[0][0]
    result = [tokenizer.get_vocabulary()[i] for i in best_seq if i != end_token and i != 0]
    return img, result
