import json
import os
from parameters.model import PredNextBERTParameters
from utils.save import load_parameters, save_parameters_json
from utils.print import print_block
from utils import VocabDict
import tensorflow as tf
import numpy as np
import pathlib
from datetime import datetime
import matplotlib.pyplot as plt

class PredNextBERT(tf.keras.Model):
    name = "PredNextBERT"
    activity_vocab_file_name = "activity_vocab.json"
    parameters_file_name = "model_params.json"

    def __init__(self,
                 activity_vocab: VocabDict,
                 parameters: PredNextBERTParameters,
                 max_input_seq_len: int = 1000,
                 ):
        super(PredNextBERT, self).__init__()
        self.activity_vocab = activity_vocab
        self.parameters = parameters
        self.tokenizer = PredictNextEncoder(
            self.parameters.num_layers,
            self.parameters.model_dim,
            self.parameters.num_heads,
             self.parameters.feed_forward_dim,
            len(self.activity_vocab), max_input_seq_len, self.parameters.dropout_rate)

        self.final_layer = tf.keras.layers.Dense(len(self.activity_vocab))

    def call(self, inp, training, combine_mask):
        # (batch_size, inp_seq_len, d_model)
        enc_output, attention_weights = self.tokenizer(
            inp, training, combine_mask)

        # (batch_size, inp_seq_len, target_vocab_size)
        final_output = self.final_layer(enc_output)

        return final_output, attention_weights

    def get_prediction_list_from_out(self, out, data):
        target = data[-1]
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        predicted = tf.math.argmax(out, axis=-1)  # (B, S)
        selected_predictions = tf.boolean_mask(
            predicted, mask)

        return selected_predictions.numpy().tolist()

    def get_target_list_from_target(self, data):
        target = data[-1]
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        selected_targets = tf.boolean_mask(
            target, mask
        )
        return selected_targets.numpy().tolist()

    def get_folder_path(self, current_file, test_accuracy, additional=""):
        saving_folder_path = os.path.join(
            pathlib.Path(current_file).parent,
            "SavedModels/%.4f_%s_%s_%s" % (test_accuracy,
                                           self.name,
                                           additional,
                                           str(datetime.now())).replacee(":", "'"),
        )
        return saving_folder_path

    def data_call(self, data, training=None):
        _, padded_data_traces, _, _, _, _ = data

        # Need mask here for calling.
        combine_mask = create_predicting_next_mask(padded_data_traces)

        out, _ = self.call(
            padded_data_traces,
            combine_mask=combine_mask,
            training=training
        )

        return out

    def get_accuracy(self, y_pred, data):
        y_true = data[-1]
        return accuracy_function(y_true, y_pred)

    def get_loss(self, loss_fn, y_pred, data):
        y_true = data[-1]
        return loss_function(y_true, y_pred)

    def get_example_input(self,):
        return {
            "inp": tf.ones((1, 1)),
            "combine_mask": None,
            "training": False
        }

    def save(self, folder_path: str):
        self.save_parameters(folder_path)
        self.save_vocabs(folder_path)
        self.save_model(folder_path)

    def save_model(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        # Save model
        model_saving_path = os.path.join(
            folder_path, "model.ckpt"
        )
        save_dict = {
            "model": self,
        }

        checkpoint = tf.train.Checkpoint(**save_dict)
        checkpoint.save(model_saving_path)
        print_block("Model saved successfully to: %s " % (folder_path))

    def save_vocabs(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        activitiy_vocab_path = os.path.join(
            folder_path, PredNextBERT.activity_vocab_file_name)
        with open(activitiy_vocab_path, 'w') as output_file:
            json.dump(self.activity_vocab.vocabs, output_file, indent='\t')

        print_block("Vocabs saved successfully to: %s " % (folder_path))

    def save_parameters(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        parameters_saving_path = os.path.join(
            folder_path, PredNextBERT.parameters_file_name
        )

        save_parameters_json(parameters_saving_path, self.parameters)

        print_block("Parameters saved successfully to: %s " % (folder_path))

    def load_model(self, folder_path: str):

        load_dict = {
            "model": self
        }

        checkpoint = tf.train.Checkpoint(
            **load_dict
        )

        checkpoint.restore(tf.train.latest_checkpoint(folder_path))

        del checkpoint

        print_block("Model loaded successfully from: %s " % (folder_path))

    @staticmethod
    def load_vocab(folder_path):
        activitiy_vocab_path = os.path.join(
            folder_path, PredNextBERT.activity_vocab_file_name)
        with open(activitiy_vocab_path, 'r') as output_file:
            vocabs = json.load(output_file)
            activity_vocab = VocabDict(vocabs)

        print_block("Vocab loaded successfully from: %s " % (folder_path))

        return activity_vocab

    @staticmethod
    def load_model_params(folder_path):
        parameters = load_parameters(
            folder_path, PredNextBERT.parameters_file_name)
        print_block("Model parameters loaded successfully from: %s " %
                    (folder_path))
        return parameters

    @staticmethod
    def load(folder_path, max_input_seq_len=1000):

        parameters_json = PredNextBERT.load_model_params(
            folder_path)

        parameters = PredNextBERTParameters(
            **parameters_json)

        activitiy_vocab = PredNextBERT.load_vocab(
            folder_path)

        model = PredNextBERT(
            activitiy_vocab,
            parameters,
            max_input_seq_len,
        )

        model.load_model(folder_path)

        return model

    def plot_attention_head(self, in_tokens, translated_tokens, attention):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = in_tokens
        ax.set_xticklabels(
            labels, rotation=90)

        labels = translated_tokens
        ax.set_yticklabels(labels)

    def plot_attention_weights(self, in_tokens, translated_tokens, layer_of_attention_heads):
        fig = plt.figure(figsize=(60, 20))

        n_l, n_h = layer_of_attention_heads.shape[:2]

        for l, attention_heads in enumerate(layer_of_attention_heads):
            n_h = len(attention_heads)
            for h, head in enumerate(attention_heads):
                ax = fig.add_subplot(n_l, n_h, (l*n_h) + h+1)
                self.plot_attention_head(in_tokens, translated_tokens, head)
                ax.set_xlabel(f'Layer {l+1} Head {h+1} ')
        plt.tight_layout()
        plt.show()

    def plot_step_attention_weight(self, step_i,  all_tokens, attentions_in_time_series, input_trace_length):
        last_step_attention = tf.concat(attentions_in_time_series[step_i], axis=0)[
            :, :, -1, :][:, :, tf.newaxis, :]
        self.plot_attention_weights(
            all_tokens[:step_i+input_trace_length], [
                all_tokens[step_i+input_trace_length]
            ],
            last_step_attention
        )

    def plot_average_attention(self, in_tokens, translated_tokens, layer_of_attention_heads):
        # The plot is of the attention when a token was generated.
        # The model didn't generate `<START>` in the output. Skip it.
        attention = tf.reduce_mean(layer_of_attention_heads, axis=[0, 1])
        ax = plt.gca()
        ax.matshow(attention)
        ax.set_xticks(range(len(in_tokens)))
        ax.set_yticks(range(len(translated_tokens)))

        labels = in_tokens
        ax.set_xticklabels(
            labels, rotation=90)
        labels = translated_tokens
        ax.set_yticklabels(labels)

    def plot_stop_mean_attention_weight(self, step_i, all_tokens, attentions_in_time_series, input_trace_length):
        last_step_attention = tf.concat(attentions_in_time_series[step_i], axis=0)[
            :, :, -1, :][:, :, tf.newaxis, :]
        self.plot_average_attention(
            all_tokens[:step_i+input_trace_length], [
                all_tokens[step_i+input_trace_length]
            ],
            last_step_attention
        )

    def predict_next(self, encoder_input, max_length=40, eos_id=1):
        # as the target is english, the first word to the transformer should be the
        # english start token.

        attentions_in_time_series = []

        for i in range(max_length):
            combined_mask = create_predicting_next_mask(encoder_input)

            # predictions.shape == (batch_size, seq_len, vocab_size)
            predictions, attention_weights = self.call(encoder_input,
                                                        False,
                                                        combined_mask)

            attentions_in_time_series.append(attention_weights)

            # select the last word from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            encoder_input = tf.concat([encoder_input, predicted_id], axis=-1)

            # return the result if the predicted_id is equal to the end token
            if predicted_id == eos_id:
                break

        # output.shape (1, tokens)

        all_predicted_tokens = self.activity_vocab.list_of_index_to_vocab(
            encoder_input.numpy()[0])

        return encoder_input, attentions_in_time_series, all_predicted_tokens



class PredictNextEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(PredictNextEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [PredictNextEncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        attention_weights = []
        for i in range(self.num_layers):
            x, attention_weight = self.enc_layers[i](x, training, mask)
            attention_weights.append(attention_weight)

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


class PredictNextEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(PredictNextEncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # (batch_size, input_seq_len, d_model)
        attn_output, attention_weight = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2, attention_weight


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def create_predicting_next_mask(inp):
    look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
    inp_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(inp_padding_mask, look_ahead_mask)
    return combined_mask


def create_predicting_next_mask(inp):
    look_ahead_mask = create_look_ahead_mask(tf.shape(inp)[1])
    inp_padding_mask = create_padding_mask(inp)
    combined_mask = tf.maximum(inp_padding_mask, look_ahead_mask)
    return combined_mask


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class BERTScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(BERTScheduler, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)