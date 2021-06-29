from parameters.dataset import BPI2012ScenarioParameters
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from utils.print import print_block
import json
from utils.file import file_exists
import os
from typing import List
from parameters.enum import BPI2012ActivityType
from datetime import timedelta
import pm4py
import pandas as pd
from utils.VocabDict import VocabDict, Constants
import numpy as np
import tensorflow as tf
import math


class BPI2012ScenarioDataset():
    pickle_df_file_name = "df.pickle"
    activity_vocab_file_name = "activity_vocab.json"
    resource_vocab_file_name = "resource_vocab.json"

    def __init__(self, parameters: BPI2012ScenarioParameters) -> None:
        super().__init__()
        self.parameters = parameters
        self.file_path = self.parameters.file_path
        self.preprocessed_folder_path = os.path.join(
            self.parameters.preprocessed_folder_path, self.info_str())

        if (not self.preprocessed_folder_path is None) and self.preprocessed_data_exist(self.preprocessed_folder_path):
            self.load_preprocessed_data()
        else:
            self.__initialise_data(
                file_path=self.file_path, include_types=self.parameters.include_types)

            if not self.preprocessed_folder_path is None:
                self.save_preprocessed_data()

    def __initialise_data(self, file_path: str, include_types: List[BPI2012ActivityType]) -> None:
        '''
        run this function if the preprocessed data doesn't exist.
        [file_path]: path of `BPI_Challenge_2012.xes`
        [include_types]: what types of activity you want to load.
        '''
        ############ load xes file and extract needed information ############
        log = pm4py.read_xes(file_path)
        flattern_log = ([{**event,
                          'caseid': trace.attributes['concept:name'],
                          'amount': trace.attributes['AMOUNT_REQ'],
                          }

                         for trace in log for event in trace])
        df = pd.DataFrame(flattern_log)
        df = df[df["lifecycle:transition"] == "COMPLETE"]
        df["org:resource"] = [
            'UNKNOWN' if math.isnan(float(r)) else r for r in df["org:resource"]]

        if not (include_types is None):
            df = df[[any(bool_set) for bool_set in zip(
                *([df["concept:name"].str.startswith(a.value) for a in include_types]))]]

        df["activity"] = df["concept:name"] + \
            "_" + df["lifecycle:transition"]

        ############ Append starting and ending time for each trace ############
        newData = []
        for case, group in df.groupby('caseid'):
            group.sort_values("time:timestamp", ascending=True, inplace=True)
            strating_time = group.iloc[0]["time:timestamp"] - \
                timedelta(microseconds=1)
            ending_time = group.iloc[-1]["time:timestamp"] + \
                timedelta(microseconds=1)
            amount = group.iloc[-1]["amount"]
            traces = group.to_dict('records')

            # Add start and end tags.
            traces.insert(
                0, {"caseid": case,
                    "time:timestamp": strating_time,
                    "activity": Constants.SOS_VOCAB,
                    "amount": amount,
                    "org:resource": Constants.SOS_VOCAB,
                    })
            traces.append(
                {"caseid": case,
                 "time:timestamp": ending_time,
                 "activity": Constants.EOS_VOCAB,
                 "amount": amount,
                 "org:resource": Constants.EOS_VOCAB,
                 })
            newData.extend(traces)

        df = pd.DataFrame(newData)
        df['activity'] = df['activity'].astype(
            'category')

        df['org:resource'] = df['org:resource'].astype('category')

        ############ generate vocabulary list ############

        self.activity_vocab = VocabDict(
            [Constants.PAD_VOCAB] + list(df['activity'].cat.categories))
        self.resource_vocab = VocabDict(
            [Constants.PAD_VOCAB] + list(df["org:resource"].cat.categories))

        ############ Create new index categorial column ############
        df['activity_idx'] = df['activity'].apply(
            lambda c: self.activity_vocab.vocab_to_index(c))

        df['resource'] = df["org:resource"].apply(
            lambda a: self.resource_vocab.vocab_to_index(a))

        ############ Create the df only consisted of trace and caseid ############
        final_df_data = []
        for caseid, group in df.groupby('caseid'):
            final_df_data.append({
                "activity": list(group['activity_idx']),
                "activity_vocab": list(group['activity']),
                "caseid": caseid,
                "amount": float(list(group['amount'])[0]),
                "resource": list(group['resource']),
                "resource_vocab": list(group['org:resource'])
            })

        ############ store data in instance ############
        self.df: pd.DataFrame = pd.DataFrame(final_df_data)
        self.df.sort_values("caseid", inplace=True)

        tag_vocabs = [self.activity_vocab.sos_vocab(
        ), self.activity_vocab.eos_vocab(), self.activity_vocab.pad_vocab()]

        self.possible_activities = [
            a for a in self.activity_vocab.vocabs if not a in tag_vocabs]
        self.possbile_resources = [
            r for r in self.resource_vocab.vocabs if not r in tag_vocabs]
        self.possible_amount = [min(self.df["amount"]), max(self.df["amount"])]

        size_times = self.parameters.sample_times

        all_lens = [len(t) for t in list(self.df["activity"])]
        current_df = dict(self.df)
        for _ in range(size_times - 1):
            self.df = self.df.append(pd.DataFrame(current_df))

        random_df = self.get_random_generated_df(all_lens * size_times)
        self.df = self.df.append(random_df)

    def longest_trace_len(self) -> int:
        return self.df.trace.map(len).max()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    def info_str(self,):
        folder_name = ""
        if self.parameters.include_types is None:
            folder_name += "All"
        else:
            folder_name += "".join(
                sorted([a.value for a in self.parameters.include_types], key=str.lower))

        folder_name += "_"

        if (self.parameters.include_complete_only):
            folder_name += "CompleteOnly"

        folder_name += f"_{self.parameters.sample_times}_times"

        return folder_name

    @staticmethod
    def get_type_folder_name(include_types: List[BPI2012ActivityType] = None):
        if include_types is None:
            return "All"

        return "".join(
            sorted([a.value for a in include_types], key=str.lower))

    @staticmethod
    def preprocessed_data_exist(preprocessed_folder_path: str):
        df_path = os.path.join(preprocessed_folder_path,
                               BPI2012ScenarioDataset.pickle_df_file_name)
        vocab_dict_path = os.path.join(
            preprocessed_folder_path, BPI2012ScenarioDataset.activity_vocab_file_name)
        return file_exists(df_path) and file_exists(vocab_dict_path)

    def store_df(self, preprocessed_folder_path: str):
        os.makedirs(preprocessed_folder_path, exist_ok=True)
        df_path = os.path.join(preprocessed_folder_path,
                               BPI2012ScenarioDataset.pickle_df_file_name)
        self.store_df_in_pickle(df_path)

    def load_df(self, preprocessed_folder_path: str):
        df_path = os.path.join(preprocessed_folder_path,
                               BPI2012ScenarioDataset.pickle_df_file_name)
        self.load_df_from_pickle(df_path)

    def store_df_in_pickle(self, path):
        self.df.to_pickle(path)

    def load_df_from_pickle(self, path):
        self.df = pd.read_pickle(path)

    def save_preprocessed_data(self):
        if self.preprocessed_folder_path is None:
            raise Exception("Preprocessed folder path can't be None")

        ############ Store df ############
        self.store_df(self.preprocessed_folder_path)

        ############ Store vocab_dict ############
        vocabs_path = os.path.join(
            self.preprocessed_folder_path, BPI2012ScenarioDataset.activity_vocab_file_name)
        with open(vocabs_path, 'w') as output_file:
            json.dump(self.activity_vocab.vocabs, output_file, indent='\t')

        ############ Store resources ############
        resources_path = os.path.join(
            self.preprocessed_folder_path, BPI2012ScenarioDataset.resource_vocab_file_name)
        with open(resources_path, 'w') as output_file:
            json.dump(self.resource_vocab.vocabs, output_file, indent='\t')

        print_block(
            "Preprocessed data saved successfully"
        )

    def load_preprocessed_data(self):
        if self.preprocessed_folder_path is None:
            raise Exception("Preprocessed folder path can't be None")

        ############ Load df ############
        self.load_df(self.preprocessed_folder_path)

        ############ load vocab_dict ############
        vocabs_path = os.path.join(
            self.preprocessed_folder_path, BPI2012ScenarioDataset.activity_vocab_file_name)
        with open(vocabs_path, 'r') as output_file:
            vocabs = json.load(output_file)
            self.activity_vocab = VocabDict(vocabs)

        ############ load resources ############
        resources_path = os.path.join(
            self.preprocessed_folder_path, BPI2012ScenarioDataset.resource_vocab_file_name)
        with open(resources_path, 'r') as output_file:
            vocabs = json.load(output_file)
            self.resource_vocab = VocabDict(vocabs)

        print_block(
            "Preprocessed data loaded successfully: %s" % (
                self.preprocessed_folder_path)
        )

    def get_sampler_from_df(self, df, seed):
        return None

    def get_train_shuffle(self):
        return True

    def get_index_ds(self):
        return tf.data.Dataset.range(len(self.df))

    def get_random_generated_df(self, all_lens):
        generated_activities = generate_random_trace(
            all_lens, possible_vocabs=self.possible_activities, sos_vocab=self.activity_vocab.sos_vocab(), eos_vocab=self.activity_vocab.eos_vocab())
        generated_activities_idx = self.activity_vocab.list_of_vocab_to_index_2d(
            generated_activities)

        generated_resources = generate_random_trace(all_lens, possible_vocabs=self.possbile_resources,
                                                    sos_vocab=self.resource_vocab.sos_vocab(), eos_vocab=self.resource_vocab.eos_vocab())
        generated_resources_idx = self.resource_vocab.list_of_vocab_to_index_2d(
            generated_resources)

        generated_amount = generated_random_amount(
            self.possible_amount, len(all_lens))

        generated_df = pd.DataFrame({})
        generated_df["activity"] = generated_activities_idx
        generated_df['activity_vocab'] = generated_activities
        generated_df['resource'] = generated_resources_idx
        generated_df['resource_vocab'] = generated_resources
        generated_df["caseid"] = "Fake"
        generated_df["amount"] = generated_amount

        return generated_df

    def collate_fn(self, idxs: List[int]):
        '''
        Return: [caseids, padded_data_traces, lengths, batch_resources, batch_amount, padded_target_traces]
        '''
        batch_df = self.df.iloc[idxs]
        caseids = list(batch_df["caseid"])
        batch_activities = list(batch_df["activity"])
        batch_resources = list(batch_df["resource"])
        batch_amount = list(batch_df["amount"])
        data_activities = [t[:-1] for t in batch_activities]
        data_resources = [r[:-1] for r in batch_resources]
        lengths = [len(t) for t in data_activities]
        target_traces = [t[1:] for t in batch_activities]
        padded_data_traces = tf.keras.preprocessing.sequence.pad_sequences(
            data_activities, padding='post', value=0)  # return will be numpy
        padded_target_traces = tf.keras.preprocessing.sequence.pad_sequences(
            target_traces, padding='post', value=0)  # return will be numpy

        padded_data_resources = tf.keras.preprocessing.sequence.pad_sequences(
            data_resources, padding='post', value=0)

        is_real = [c != "Fake" for c in caseids]

        real_trace = tf.keras.preprocessing.sequence.pad_sequences(
            [[1]*l if r else [0] * l for l, r in zip(lengths, is_real)], value=-1, padding="post")  # numpy array

        # Don't get loss for the first 4 output.
        # real_trace[:, :4] = -1
        # real_trace = np.ones_like(real_trace)

        return caseids, padded_data_traces, lengths, padded_data_resources, batch_amount, padded_target_traces, real_trace


def generate_random_trace(lengths, possible_vocabs, sos_vocab, eos_vocab):
    generated = []

    for l in lengths:
        l = l - 2
        generated.append(
            [sos_vocab] + np.random.choice(possible_vocabs, l).tolist() + [eos_vocab])
    return generated


def generated_random_amount(possible_amount, size):
    return np.random.randint(possible_amount[0], possible_amount[1], size).astype(np.float64).tolist()
