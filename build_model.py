import tensorflow as tf
from bert_layer import BertLayer

# Build model
def build_model(max_seq_length, num_classes):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    # just one dense layer on top of BERT proved to perform the best
    pred = tf.keras.layers.Dense(num_classes, activation="sigmoid")(bert_output)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    return model
