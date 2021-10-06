import time
import numpy as np
import pandas as pd
import textdistance
import tensorflow as tf

from enum import Enum
from utils.print import print_block
from utils.bpi2012 import remove_tags_for_seq
from tensorflow.python.ops.random_ops import categorical


class FeatureType(Enum):
    Categorical = "Categorical"
    Numerical = "Numerical"


class DiCE4EL_BPI2012():

    def __init__(self,
                 activity_vocab,
                 resource_vocab,
                 possible_amount,
                 possible_activities,
                 possible_resources,
                 pred_model,
                 train_df,
                 activity_milestones=[
                     "A_SUBMITTED_COMPLETE",
                     "A_PARTLYSUBMITTED_COMPLETE",
                     "A_PREACCEPTED_COMPLETE",
                     "A_ACCEPTED_COMPLETE",
                     "A_FINALIZED_COMPLETE",
                     "O_SELECTED_COMPLETE",
                     "O_CREATED_COMPLETE",
                     "O_SENT_COMPLETE",
                     "O_SENT_BACK_COMPLETE",
                     "A_APPROVED_COMPLETE",
                     "A_ACTIVATED_COMPLETE",
                     "A_REGISTERED_COMPLETE"
                 ],
                 no_need_tags=['<EOS>', '<SOS>', '<PAD>'],
                 ):

        self.pred_model = pred_model

        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab

        self.possible_activities = possible_activities
        self.possible_amount = possible_amount
        self.possible_resources = possible_resources

        # Searching needed
        self.train_df = train_df
        self.activity_milestone = activity_milestones
        self.no_need_tags = no_need_tags

        self.return_features = ['activity', 'activity_vocab', 'resource', 'resource_vocab',
                                'amount', 'predicted_vocab', 'activity_sparcity']

    def search(self, activities, desired):
        # We have to pick one "Amount" & "Replac_Amount" combination for this algo.
        # => Both should be "None"

        # only keep the activties that are not milestones in the trace.
        milestone_trace = [
            a for a in activities if a in self.activity_milestone]

        # Find the cases containing milestone_filtered in training set.
        query_df = self.train_df[[all(
            [v in t['activity_vocab'] for v in milestone_trace]) for t in self.train_df.iloc]]

        # Find cases containing desired df (cases with same milestone).
        desired_df = query_df[[
            desired in v for v in query_df['activity_vocab']]]

        if (len(desired_df) <= 0):
            raise Exception("Not matches activities found in trainig set")

        # Remove ground truth tails, so we can use model to make prediction to see if the prediction is the same as the ground truth.
        for idx in list(desired_df.index):
            desired_idx = desired_df.loc[idx]['activity_vocab'].index(desired)

            for col in ['activity', 'activity_vocab', 'resource', 'resource_vocab']:
                desired_df.at[idx,
                              col] = desired_df.loc[idx][col][:desired_idx]

        desired_df = pd.DataFrame(desired_df)

        # Make actual prediction to find the actual counterfactual.
        all_predicted_vocabs = []
        all_predicted_value = []
        for idx in range(len(desired_df)):
            ex_activity = tf.constant(
                [desired_df.iloc[idx]['activity']], dtype=tf.float32)
            ex_resource = tf.constant(
                [desired_df.iloc[idx]['resource']], dtype=tf.float32)
            ex_amount = tf.constant(
                [desired_df.iloc[idx]['amount']], dtype=tf.float32)
            out, _ = self.pred_model(
                ex_activity, ex_resource, ex_amount, training=False)
            out = tf.nn.softmax(out, axis=-1)
            pred_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]
            predicted_vocab = self.pred_model.activity_vocab.index_to_vocab(
                pred_idx)
            all_predicted_vocabs.append(predicted_vocab)
            all_predicted_value.append(out[:, -1, pred_idx].numpy()[0])

        desired_df['predicted_vocab'] = all_predicted_vocabs
        desired_df['predicted_value'] = all_predicted_value

        # Get the actual counterfactual..
        desired_df = desired_df[desired_df['predicted_vocab'] == desired]

        if (len(desired_df) <= 0):
            raise Exception("Not matches activities found in trainig set")

        # Only keep the one with minimum distance.
        desired_df['activity_sparcity'] = [textdistance.levenshtein.distance(
            activities, a) for a in desired_df['activity_vocab']]
        desired_df = desired_df[desired_df.activity_sparcity ==
                   desired_df.activity_sparcity.min()]

        return desired_df

    def min_max_scale_amount(self, input_amount, inverse=False):
        '''
        Min-max scale the amount.
        '''
        min_a = self.possible_amount[0]
        max_a = self.possible_amount[1]
        min_max_range = (max_a - min_a)
        if inverse:
            return (input_amount * min_max_range) + min_a
        else:
            return input_amount - min_a / min_max_range

    def get_pred_model_ouptut(self, data_input):
        '''
        Get pred model output.
        '''
        traces, resources, amount = data_input
        out, _ = self.pred_model(traces, resources, amount, training=False)
        predicted_idx = tf.argmax(out[:, -1, :], axis=-1).numpy().tolist()
        return out[:, -1, :],  predicted_idx

    def generate_prediction_for_df(self, df) :
        all_predicted_vocabs = []
        all_predicted_value = []

        for idx in range(len(df)):
            ex_activity = tf.constant(
                [df.iloc[idx]['activity']], dtype=tf.float32)
            ex_resource = tf.constant(
                [df.iloc[idx]['resource']], dtype=tf.float32)
            ex_amount = tf.constant(
                [df.iloc[idx]['amount']], dtype=tf.float32)
            out, _ = self.pred_model(
                ex_activity, ex_resource, ex_amount, training=False)
            out = tf.nn.softmax(out, axis=-1)
            pred_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]
            predicted_vocab = self.pred_model.activity_vocab.index_to_vocab(
                pred_idx)
            all_predicted_vocabs.append(predicted_vocab)
            all_predicted_value.append(
                out[:, -1, pred_idx].numpy()[0])

        return all_predicted_vocabs, all_predicted_value

    def generate_prediction_for_df_and_variable_amount(self, df, amonut_v) :
        all_predicted_vocabs = []
        all_predicted_value = []
        all_predicted_idx = []

        for idx in range(len(df)):
            ex_activity = tf.constant(
                [df.iloc[idx]['activity']], dtype=tf.float32)
            ex_resource = tf.constant(
                [df.iloc[idx]['resource']], dtype=tf.float32)
            out, _ = self.pred_model(
                ex_activity, ex_resource, amonut_v[idx], training=False)

            out = tf.nn.softmax(out, axis=-1)
            pred_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]
            predicted_vocab = self.pred_model.activity_vocab.index_to_vocab(
                pred_idx)
            all_predicted_idx.append(pred_idx)
            all_predicted_vocabs.append(predicted_vocab)
            all_predicted_value.append(
                out[:, -1, :][0])

        return all_predicted_idx, all_predicted_vocabs, all_predicted_value

    def generate_counterfactual(
        self,
        amount_input,
        idx_activities,
        idx_resources,
        desired_vocab,
        class_using_hinge_loss=True,
        use_clipping=True,
        class_loss_weight=1.0,
        distance_loss_weight=1e-8,
        verbose_freq=20,
        max_iter=200,
        lr=0.05,
    ):
        # Recording time consuming.
        start_at = time.time()

        # Setting up desired.
        desired_vocab_idx = self.activity_vocab.vocab_to_index(desired_vocab)
        self.desired_vocab = desired_vocab
        self.desired_vocab_idx = desired_vocab_idx

        orignin_activity_input = tf.constant( idx_activities,dtype=tf.float32)
        origin_resource_input = tf.constant( idx_resources, dtype=tf.float32)
        origin_amount_input = tf.constant(amount_input)

        ## Check current prediction
        origin_out, _ = self.pred_model(orignin_activity_input, origin_resource_input, origin_amount_input, training=False)
        origin_out = tf.nn.softmax(origin_out, axis=-1)
        init_pred_idx = tf.argmax(origin_out[:, -1, :], axis=-1).numpy()[0]
        init_pred_vocab = self.activity_vocab.index_to_vocab(
            init_pred_idx)

        print_block(f"{init_pred_vocab} ====> {desired_vocab}", "Generating counterfactaul...")

        if (init_pred_idx == desired_vocab_idx):
            print_block("The prediction is already desired vocab")
            return
            
        # Search for the counterfactuals in dataset.
        desired_df = self.search(self.activity_vocab.list_of_index_to_vocab(idx_activities.tolist()[0]), desired_vocab)

        print_block(f"Found {len(desired_df)} potentail counterfactuals in training set.", "Searching Done")

        ### Replace by input account to see if any is good => then we return those.
        desired_df['amount'] = [amount_input] * len(desired_df)

        self.desired_df = desired_df

        same_amount_all_predicted_vocabs,  same_amount_all_predicted_value = self.generate_prediction_for_df(desired_df)

        desired_df['same_amount_predicted_vocab'] = same_amount_all_predicted_vocabs
        desired_df['same_amount_predicted_value'] = same_amount_all_predicted_value

        # Filter the vocab again.
        same_amount_desired_df = desired_df[desired_df['same_amount_predicted_vocab'] == desired_vocab]

        if len(same_amount_desired_df) > 0:
            print_block(f"Found {len(same_amount_desired_df)} cases.", "SAME AMOUNT COUNTERFACTUALS")
            return same_amount_desired_df[self.return_features]

        print_block(
            f"Number of Cases remain after setting the same amount as input: {len(desired_df)}.")

        # if no counterfactual exist after we replace all the amount by our amount, we perform updating throught Grdient Decent. 

        print_block("Start updating amonut to find counterfactuals.")

        #### *** Length of the activities are different, so we have to make prediction for each.

        cf_amounts = tf.Variable(np.ones((len(desired_df), 1)) * amount_input, dtype=tf.float32)
        amounts_backup = cf_amounts.numpy()

        for i in range(max_iter):

            optim = tf.keras.optimizers.Adam(learning_rate=lr)

            with tf.GradientTape() as tape:
                # Get prediction from cf
                # Maybe we can convert it to the original input here

                cf_pred_idx, _, cf_predicted_value = self.generate_prediction_for_df_and_variable_amount(
                    desired_df, amonut_v= cf_amounts)

                # Distance loss
                distance_loss = self.min_max_scale_amount(
                    tf.pow(cf_amounts - amounts_backup, 2))

                # Trying to use another loss.
                if class_using_hinge_loss:
                    class_loss = tf.keras.metrics.hinge(
                        [1] * len(desired_df), cf_predicted_value)
                else:
                    self.desired_df = desired_df
                    self.desired_vocab_idx = desired_vocab_idx
                    self.cf_predicted_value = cf_predicted_value
                    class_loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_true= np.array([desired_vocab_idx] * len(desired_df)).reshape((1, -1)),
                        y_pred= tf.stack(cf_predicted_value)
                    )

                # Get total loss
                # the category loss need to prevent the updated cf so different from the original one.
                loss = (class_loss_weight * class_loss) + (distance_loss * distance_loss_weight)

            if (i != 0) and i % verbose_freq == 0:
                print_block(f"Total [{loss.numpy().flatten()[0]:.2f}] | " +
                            f"Class [{class_loss.numpy().flatten()[0]:.2f}] | " +
                            f"Distance [{distance_loss.numpy().flatten()[0]:.2f}]",
                            f"Step {i} Loss")

            grad = tape.gradient(loss,  [cf_amounts])

            self.grad = grad

            optim.apply_gradients(zip(grad, [cf_amounts]))

            # Clipping the value for next round (This step has a risk to cause the nn can't find the local minimum.)
            if use_clipping:
                cf_amounts.assign(tf.clip_by_value(cf_amounts, self.possible_amount[0], self.possible_amount[1]))


            # Make prediction again.

            temp_pred_idxes, _, _ = self.generate_prediction_for_df_and_variable_amount(
                    desired_df, amonut_v= cf_amounts)

            if (i != 0) and i % verbose_freq == 0:
                ### Verbose.
                cf_pred_vocab = self.activity_vocab.list_of_index_to_vocab(
                    cf_pred_idx)
                temp_pred_vocab = self.activity_vocab.list_of_index_to_vocab(
                    temp_pred_idxes)
                print_block(f"Invalid: {cf_pred_vocab} ({cf_pred_idx}) | Valid: {temp_pred_vocab} ({temp_pred_idxes})",
                            f"Step {i} CF predicted")

            
            if (all([ i == desired_vocab_idx for i in temp_pred_idxes])):
                ## If all match.
                # Replace amount and give desired_vocab
                temp_pred_vocab = self.activity_vocab.list_of_index_to_vocab(
                    temp_pred_idxes)
                desired_df['amount'] = cf_amounts.numpy().tolist()
                desired_df['predicted_vocab'] = temp_pred_vocab
                print_block(f"Running time: {time.time() - start_at:.2f}",
                                f"! Counterfactual found in step [{i+1}] \U0001F917 !")

                return desired_df[self.return_features]


        temp_pred_vocab = self.activity_vocab.list_of_index_to_vocab(
                    temp_pred_idxes)
        desired_df['amount'] = cf_amounts.numpy().tolist()
        desired_df['predicted_vocab'] = temp_pred_vocab


        if (temp_pred_idxes.count(desired_vocab_idx) > 0):
            print_block(f"Running time: {time.time() - start_at:.2f}",
                                f"| {len(desired_df[desired_df.predicted_vocab == desired_vocab])} Counterfactuals found in \U0001F917 |")
            return desired_df[desired_df.predicted_vocab == desired_vocab][self.return_features]
        

        return desired_df[self.return_features]