import json
from tensorflow.python import keras

from tensorflow.python.keras import models
from tensorflow.python.keras.api._v2.keras import layers
from tensorflow.python.keras.layers import dense_attention
from tensorflow.python.keras.layers.core import Dense
from utils.print import print_block
from utils.save import load_parameters, save_parameters_json
from datetime import datetime
import os
import pathlib
from model.ControllerModel import ControllerModel
import tensorflow as tf
from utils.VocabDict import Constants, VocabDict
import numpy as np
from typing import List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from parameters.model import LSTMPredNextAmountSpecificModelParameters


class LSTMPredNextAmountSpecificModel(ControllerModel):
    name = "LSTMPredNextAmountSpecificModel"
    activity_vocab_file_name = "activity_vocab.json"
    resource_vocab_file_name = "resource_vocab.json"
    parameters_file_name = "model_params.json"

    def __init__(self,
                 activity_vocab: VocabDict,
                 resource_vocab: VocabDict,
                 parameters: LSTMPredNextAmountSpecificModelParameters,
                 ):
        super().__init__()
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.parameters = parameters

        self.activity_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.activity_vocab),
            output_dim=self.parameters.activity_embedding_dim,
            mask_zero=True,
        )

        self.resource_embedding = tf.keras.layers.Embedding(
            input_dim=len(self.resource_vocab),
            output_dim=self.parameters.resource_embedding_dim,
            mask_zero=True
        )

        self.activity_lstm = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.activity_lstm_sec = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.resource_lstm = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.resource_lstm_sec = tf.keras.layers.LSTM(
            self.parameters.lstm_hidden,
            return_sequences=True,
            return_state=True,
        )

        self.amount_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(self.parameters.dense_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.parameters.dropout),
                tf.keras.layers.Dense(self.parameters.dense_dim)
            ]
        )

        self.out_net = tf.keras.models.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.parameters.dropout),
                tf.keras.layers.Dense(self.parameters.dense_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(self.parameters.dropout),
                tf.keras.layers.Dense(len(activity_vocab)),
            ]
        )

    def call(self, activities, resources, amount, init_state=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        if len(activities.shape) == 3:
            activity_emb_out = tf.matmul(
                activities, tf.squeeze(
                    tf.stack(self.activity_embedding.get_weights(), axis=0))
            )
            resource_emb_out = tf.matmul(
                resources, tf.squeeze(
                    tf.stack(self.resource_embedding.get_weights(), axis=0))
            )
            mask = None
        else:
            activity_emb_out = self.activity_embedding(
                activities,
                training=training
            )
            resource_emb_out = self.resource_embedding(
                resources,
                training=training
            )
            mask = self.activity_embedding.compute_mask(activities)

        max_length = activity_emb_out.shape[1]

        activity_lstm_out, a_h_out, a_c_out = self.activity_lstm(
            activity_emb_out, training=training, mask=mask, initial_state=init_state[0] if init_state else None)

        activity_lstm_out_sec, a_h_out_sec, a_c_out_sec = self.activity_lstm_sec(
            activity_lstm_out, training=training, mask=mask, initial_state=init_state[1] if init_state else None)

        resources_lstm_out, r_h_out, r_c_out = self.resource_lstm(
            resource_emb_out, training=training, mask=mask, initial_state=init_state[2] if init_state else None)

        resources_lstm_out_sec, r_h_out_sec, r_c_out_sec = self.resource_lstm_sec(
            resources_lstm_out, training=training, mask=mask, initial_state=init_state[3] if init_state else None)

        amount_to_concate = tf.repeat(
            self.amount_net(tf.constant(amount, dtype=tf.float32)[:, tf.newaxis])[
                :, tf.newaxis, :],
            max_length,
            axis=1
        )

        concat_out = tf.concat(
            [
                activity_lstm_out_sec,
                resources_lstm_out_sec,
                amount_to_concate
            ],
            axis=-1
        )

        out = self.out_net(concat_out, training=training)

        # return out, [(a_h_out, a_c_out), (r_h_out, r_c_out)]
        return out, [(a_h_out, a_c_out), (a_h_out_sec, a_c_out_sec), (r_h_out, r_c_out), (r_h_out_sec, r_c_out_sec)]

    def data_call(self, data, training=None):
        '''
        Expecting data has this order [caseids, padded_data_traces, lengths, padded_data_resources, batch_amount, padded_target_traces]
        '''
        _, padded_data_traces, _, padded_data_resources, amount, _ = data

        out, _ = self.call(
            padded_data_traces,
            padded_data_resources,
            amount,
            training=training
        )
        return out

    def get_accuracy(self, y_pred, data):
        '''
        Use argmax to get the final output, and get accuracy from it.
        [out]: output of model.
        [target]: target of input data.
        --------------
        return: accuracy value
        '''
        y_true = data[-1]
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        pred_value = tf.math.argmax(y_pred, axis=-1)
        accuracy = tf.math.reduce_mean(
            tf.cast(
                tf.boolean_mask(
                    y_true == pred_value, y_true != 0
                ),
                dtype=tf.float32
            )).numpy()

        return accuracy

    def get_loss(self, loss_fn: callable, y_pred, data):
        '''
        [loss_fn]: loss function to compute the loss.\n
        [out]: output of the model\n
        [target]: target of input data.\n

        ---------------------
        return: loss value
        '''
        y_true = data[-1]
        self.data = data
        self.y_pred = y_pred
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        loss_all = loss_fn(y_true=y_true, y_pred=y_pred)
        loss_all = loss_all * tf.cast(y_true != 0, dtype=tf.float32)
        loss = tf.reduce_mean(loss_all)
        return loss

    def get_labels(self):
        return self.activity_vocab.vocabs.keys()

    def predict_next(self, input: tf.Tensor, lengths: np.array, initial_state=None, use_argmax: bool = False, **kwargs):
        '''
        Predict next activity.
        [input]: input traces.
        [lengths]: length of traces.
        [previous_hidden_state]: hidden state in last time step, should be (h_, c_)
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: tuple(output, (h_out, c_out)).
        '''
        # batch_size = input.shape[0]  # (B, S)
        # def call(self, inputs, input_resources, amount, init_state=None, training=None):

        out, hidden_out = self.call(
            input, initial_state=initial_state, training=False, **kwargs)  # (B, S, vocab_size)

        ############ Get next activity ############
        # Get the last output from each seq
        # len - 1 to get the index,
        # a len == 80 seq, will only have index 79 as the last output (from the 79 input)

        # Get the output of last timestamp
        final_index = lengths - 1
        out = tf.gather(out, final_index, axis=1)
        # out = out[np.arange(batch_size), final_index, :]  # (B, Vocab)

        if (use_argmax):
            ############ Get the one with largest possibility ############
            out = tf.math.argmax(out, axis=-1)  # (B)
            # TODO: Testing value, need to delete
        else:
            ############ Sample from distribution ############
            out = tf.random.categorical(out, 1).squeeze(
                1)  # .squeeze()  # (B)

        return out, hidden_out

    def predict_next_n(self, input: tf.Tensor, n: int, lengths: np.array = None, use_argmax: bool = False, **kwargs) -> List[List[int]]:
        '''
        peform prediction n times.\n
        [input]: input traces
        [n]: number of steps to peform prediction.
        [lengths]: lengths of traces
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        -------------
        return: predicted list.
        '''
        ############ Unpadded input to get current taces ############
        predicted_list = [[i.numpy() for i in l if i != 0] for l in input]

        ############ Initialise hidden state ############
        hidden_state = None
        for i in range(n):
            ############ Predict############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        initial_state=hidden_state, use_argmax=use_argmax, **kwargs)

            ############ Add predicted to current traces ############
            predicted_list = [u + [p.numpy()[0]]
                              for u, p in zip(predicted_list, predicted)]

            ############ Prepare for next step #########################################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            # And, we only use last step and the hidden state for predicting next.
            ############################################################################################################
            # input = tf.expand_dims(predicted, axis=-1)
            lengths = np.ones_like(lengths)

        return predicted_list

    def predict_next_till_eos(self, input: tf.Tensor, lengths: np.array, eos_idx: int, use_argmax: bool = False, max_predicted_lengths=1000, **kwargs) -> List[List[int]]:
        '''
        pefrom predicting till <EOS> token show up.\n
        [input]: input traces
        [lengths]: lengths of traces
        [eos_idx]: index of <EOS> token
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.
        -------------
        return: predicted list.
        '''

        ############ List for input data ############
        input_list = [[i.numpy() for i in l if i != 0] for l in input]

        ############ List that prediction has been finished ############
        predicted_list = [None] * len(input_list)

        ############ Initialise hidden state ############
        hidden_state = None
        while len(input_list) > 0:
            ############ Predict ############
            predicted, hidden_state = self.predict_next(input=input, lengths=lengths,
                                                        previous_hidden_state=hidden_state, use_argmax=use_argmax, **kwargs)

            ############ Check if it's 0-d tensor ############
            if (predicted.size() == ()):
                predicted = predicted.unsqueeze(0)

            for idx,  (il, p) in enumerate(zip(input_list, predicted)):
                ############ Append predicted value ############
                p_v = p.numpy()
                input_list[idx] = il + [p_v]

                if (p_v == eos_idx or len(input_list[idx]) > max_predicted_lengths):
                    ############ Create index mapper (Mapping the input_list to predicted_list) ############
                    idx_mapper = [idx for idx, pl in enumerate(
                        predicted_list) if pl is None]

                    ############ Assign to predicted_list (Remove from input list) ############
                    idx_in_predicted_list = idx_mapper[idx]
                    predicted_list[idx_in_predicted_list] = input_list.pop(idx)

                    batch_size = len(predicted)
                    ############ Remove instance from the lengths ############
                    lengths = lengths[np.arange(batch_size) != idx]

                    ############ Remove instance from next input ############
                    predicted = predicted[np.arange(batch_size) != idx, ]

                    ############ Remove the hidden state to enable next inter ############
                    # h0 = hidden_state[0][:, np.arange(batch_size) != idx, :]
                    # c0 = hidden_state[1][:, np.arange(batch_size) != idx, :]

                    # TODO: Have to check the size with this one
                    h0 = tf.boolean_mask(
                        hidden_state[0],  np.arange(batch_size) != idx, axis=1)
                    c0 = tf.boolean_mask(
                        hidden_state[1],  np.arange(batch_size) != idx, axis=1)
                    hidden_state = (h0, c0)

                    if (len(predicted) == 0 and len(input_list) == 0):
                        break

            ############################################################
            # Assign for the next loop input, since tensor use reference, we won't use too much memory for it.
            ############################################################
            input = predicted.unsqueeze(-1)
            lengths = np.ones_like(lengths)

        return predicted_list

    def predict(
        self,
        input: tf.Tensor,
        lengths: np.array = None,
        n_steps: int = None,
        use_argmax=False,
        max_predicted_lengths=50,
        **kwargs
    ) -> List[List[int]]:
        '''
        [input]: tensor to predict\n
        [lengths]: lengths of input\n
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n
        [max_predicted_lengths]: used to restricted the maximum step when n_step is None.

        ----------------------------
        return: predicted list.

        '''
        if not n_steps is None:
            ######### Predict for next n activities #########
            predicted_list = self.predict_next_n(
                input=input, lengths=lengths, n=n_steps, use_argmax=use_argmax, **kwargs
            )

        else:
            ######### Predict till <EOS> token #########
            '''
            This method has the risk of causing infinite loop,
            `max_predicted_lengths` is used for restricting this behaviour.
            '''
            predicted_list = self.predict_next_till_eos(
                input=input,
                lengths=lengths,
                eos_idx=self.activity_vocab.vocab_to_index(
                    Constants.EOS_VOCAB
                ),
                use_argmax=use_argmax,
                max_predicted_lengths=max_predicted_lengths,
                **kwargs
            )

        return predicted_list

    def predicting_from_list_of_idx_trace(
        self, data: List[List[int]], n_steps: int = None, use_argmax=False
    ):
        '''
        [data]: 2D list of token indexs.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of token indexs.
        '''

        ######### To sort the input by lengths and get lengths #########
        _, data, lengths = self.activity_vocab.tranform_to_input_data_from_seq_idx_with_caseid(
            data)

        ######### Predict #########
        predicted_list = self.predict(
            input=data,
            lengths=lengths, n_steps=n_steps, use_argmax=use_argmax
        )

        return predicted_list

    def predicting_from_list_of_vacab_trace(
        self, data: List[List[str]], n_steps: int = None, use_argmax=False
    ):
        '''
        [data]: 2D list of tokens.
        [n_step]: how many steps will be predicted. If n_step == None, model will
        repeat to predict till <EOS> token show.\n
        [use_argmax]: \n
        - True -> Select the vocab with largest possibility.\n
        - False -> the next prediction will be sampled from the distribution.\n

        ----------------
        return predited 2d list of tokens.
        '''

        ######### Transform to index #########
        data = [self.activity_vocab.list_of_vocab_to_index(l) for l in data]

        ######### Predict #########
        predicted_list = self.predicting_from_list_of_idx_trace(
            data=data, n_steps=n_steps, use_argmax=use_argmax
        )

        ######### Tranform back to vocab #########
        predicted_list = [
            self.activity_vocab.list_of_index_to_vocab(l) for l in predicted_list
        ]

        return predicted_list

    def get_prediction_list_from_out(self, out, data):
        mask = self.generate_mask(data[-1])
        predicted = tf.math.argmax(out, axis=-1)  # (B, S)
        selected_predictions = tf.boolean_mask(
            predicted, mask)

        return selected_predictions.numpy().tolist()

    def get_target_list_from_target(self, data):
        target = data[-1]
        mask = self.generate_mask(target)
        selected_targets = tf.boolean_mask(
            target, mask
        )
        return selected_targets.numpy().tolist()

    def generate_mask(self, target):
        return target != 0

    def has_embedding_layer(self,):
        return True

    def calculate_embedding_distance_probs(self,):
        # Initialise stage
        embedding_matrix = self.activity_embedding.get_weights()[0]
        ordered_vocabs = []
        for i in range(len(self.activity_vocab)):
            ordered_vocabs.append(self.activity_vocab.index_to_vocab(i))

        all_probs = []
        for i in range(embedding_matrix.shape[0]):
            input_point = embedding_matrix[i]
            distance_list = [np.linalg.norm(
                embedding_matrix[m, :] - input_point) for m in range(embedding_matrix.shape[0])]
            distance_list_reverse = 1 / np.exp(distance_list)
            prob_list = distance_list_reverse / sum(distance_list_reverse)
            all_probs.append(prob_list)

        self.embedding_distance_probs = np.array(all_probs)

    def plot_activity_embedding_layer_pca(self):
        embedding_matrix = self.activity_embedding.get_weights()[0]
        ordered_vocabs = []
        for i in range(len(self.activity_vocab)):
            ordered_vocabs.append(self.activity_vocab.index_to_vocab(i))

        pca = PCA(n_components=2)
        embedding_pca = pca.fit_transform(embedding_matrix)
        fig, ax = plt.subplots(figsize=(15, 15))
        for i in range(len(ordered_vocabs)):
            ax.scatter(embedding_pca[i, 0], embedding_pca[i, 1])
            ax.annotate(
                ordered_vocabs[i], (embedding_pca[i, 0], embedding_pca[i, 1]))

    def plot_resource_embedding_layer_pca(self):
        embedding_matrix = self.resource_embedding.get_weights()[0]
        ordered_resources = []
        for i in range(len(self.resource_vocab)):
            ordered_resources.append(self.resource_vocab.index_to_vocab(i))

        pca = PCA(n_components=2)
        embedding_pca = pca.fit_transform(embedding_matrix)
        fig, ax = plt.subplots(figsize=(15, 15))
        for i in range(len(ordered_resources)):
            ax.scatter(embedding_pca[i, 0], embedding_pca[i, 1])
            ax.annotate(
                ordered_resources[i], (embedding_pca[i, 0], embedding_pca[i, 1]))

    def show_model_info(self):
        out = self.call(tf.ones((1, 1)), tf.ones(
            (1, 1)), [0.0], training=False)
        print(out)
        self.summary()

    def get_folder_path(self, current_file, test_accuracy, additional=""):
        saving_folder_path = os.path.join(
            pathlib.Path(current_file).parent,
            "SavedModels/%.4f_%s_%s_%s" % (test_accuracy,
                                           self.name,
                                           additional,
                                           str(datetime.now())),
        )
        return saving_folder_path

    def get_example_input(self,):
        return {
            "activities": tf.ones((1, 1)),
            "resources": tf.ones((1, 1)),
            "amount": [0.0],
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
            folder_path, LSTMPredNextAmountSpecificModel.activity_vocab_file_name)
        with open(activitiy_vocab_path, 'w') as output_file:
            json.dump(self.activity_vocab.vocabs, output_file, indent='\t')

        resource_vocab_path = os.path.join(
            folder_path, LSTMPredNextAmountSpecificModel.resource_vocab_file_name)
        with open(resource_vocab_path, 'w') as output_file:
            json.dump(self.resource_vocab.vocabs, output_file, indent='\t')

        print_block("Vocabs saved successfully to: %s " % (folder_path))

    def save_parameters(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        parameters_saving_path = os.path.join(
            folder_path, LSTMPredNextAmountSpecificModel.parameters_file_name
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
            folder_path, LSTMPredNextAmountSpecificModel.activity_vocab_file_name)
        with open(activitiy_vocab_path, 'r') as output_file:
            vocabs = json.load(output_file)
            activity_vocab = VocabDict(vocabs)

        resource_vocab_path = os.path.join(
            folder_path, LSTMPredNextAmountSpecificModel.resource_vocab_file_name)
        with open(resource_vocab_path, 'r') as output_file:
            vocabs = json.load(output_file)
            resource_vocab = VocabDict(vocabs)

        print_block("Vocab loaded successfully from: %s " % (folder_path))

        return activity_vocab, resource_vocab

    @staticmethod
    def load_model_params(folder_path):
        parameters = load_parameters(
            folder_path, LSTMPredNextAmountSpecificModel.parameters_file_name)
        print_block("Model parameters loaded successfully from: %s " %
                    (folder_path))
        return parameters

    @staticmethod
    def load(folder_path):

        parameters_json = LSTMPredNextAmountSpecificModel.load_model_params(
            folder_path)

        parameters = LSTMPredNextAmountSpecificModelParameters(
            **parameters_json)

        activitiy_vocab, resource_vocab = LSTMPredNextAmountSpecificModel.load_vocab(
            folder_path)

        model = LSTMPredNextAmountSpecificModel(
            activitiy_vocab,
            resource_vocab,
            parameters
        )

        model.load_model(folder_path)

        return model
