import numpy as np
import pandas as pd
import tensorflow as tf
from utils.print import print_block

def get_trace_with_idx(df, id):
    return df[[id in t for t in df["trace"]]]

def get_longest_trace_row(df):
    max_len_idx = np.argmax([len(t) for t in df['trace']])
    max_len_row = df.iloc[max_len_idx: max_len_idx+1]
    return max_len_row

def remove_trail_steps(activities_2d, resources_2d, last_n_steps: int):
    example_idx_activities = np.array([activities_2d[0][:-last_n_steps]])
    example_idx_resources = np.array([resources_2d[0][:-last_n_steps]])
    return example_idx_activities, example_idx_resources

def print_model_prediction_result(model, activities, resources, amount):
    out, _ = model(activities, resources, amount, training=False)
    out = tf.nn.softmax(out, axis=-1)
    predicted_vocab_distributions = tf.gather(
        out, activities.shape[-1]-1, axis=1)
    predicted_vocab_distributions_df = pd.DataFrame(
        predicted_vocab_distributions.numpy().tolist(), columns=model.activity_vocab.vocabs)
    max_arg = tf.math.argmax(predicted_vocab_distributions, axis=-1).numpy()[0]
    max_prob_vocab = model.activity_vocab.index_to_vocab(max_arg)
    print_block("Predicted activity with highest probability (%.2f) is \"%s\"" % (
        predicted_vocab_distributions[0][max_arg].numpy(), max_prob_vocab), "Predict result", num_marks=40)
    print("\n\n")
    print(predicted_vocab_distributions_df.iloc[0])
    return predicted_vocab_distributions_df

def remove_tags_for_query_instance(activities, resources, rm_idx_activity, rm_idx_resource):
    activities_rm = [i for i in activities[0] if not i in rm_idx_activity]
    resources_rm = [i for i in resources[0] if not i in rm_idx_resource]
    return activities_rm, resources_rm

def remove_tags_for_seq(input_seq, tags_to_remove):
    return [i for i in input_seq if not i in tags_to_remove]
