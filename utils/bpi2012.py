import numpy as np
import pandas as pd
import tensorflow as tf
from utils.print import print_block
from typing import List

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


def generate_fake_df(size: int, activity_feature_names: List[str], resource_feature_names: List[str], possible_activities: List, possible_resources: List, possbile_amount: List, trace_len: int):
    '''
    Generate a fake df to feed in DiCE Data since the DiCE doesn't support
    "features_to_vary" argument when using "features" rather than "dataframe"
    to create dice_ml.Data instance.
    '''
    if len(possbile_amount) != 2:
        raise ValueError(
            'possbile_amount should have length of 2 => [ min_, max_ ]')

    fake_df = pd.DataFrame([])

    # fake activities
    for i in range(trace_len):
        fake_df[activity_feature_names[i]] = np.random.choice(
            possible_activities, size)

    # fake resources
    for i in range(trace_len):
        fake_df[resource_feature_names[i]] = np.random.choice(
            possible_resources, size)

    # fake amount
    # fake_df['amount'] = np.random.uniform(
    #     possbile_amount[0], possbile_amount[1], (size,))

    fake_df['amount'] = np.random.randint(
        possbile_amount[0], possbile_amount[1], (size,))

    # fake label
    fake_df['predicted'] = np.random.choice([0, 1], size)

    return fake_df
