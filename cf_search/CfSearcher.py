import tensorflow as tf
import pandas as pd

class CfSearcher(object):
    def __init__(self, training_df, pred_model) -> None:
        super().__init__()

        self.training_df = training_df
        self.pred_model = pred_model

        self.milestones = [
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
        ]

    def search(self, activities, desired, amount=None):

        milestone_filtered = [
            a for a in activities if a in self.milestones]

        query_df = self.training_df[[all(
            [v in t['activity_vocab'] for v in milestone_filtered]) for t in self.training_df.iloc]]

        if not amount is None:
            query_df = query_df[query_df['amount']==amount]

        # Find with desired
        desired_df = query_df[[ desired in v for v in query_df['activity_vocab']]]

        if (len(desired_df) <= 0):
            raise Exception("Not matches found in trainig set")


        for idx in list(desired_df.index):
            desired_idx = desired_df.loc[idx]['activity_vocab'].index(desired)

            for col in ['activity', 'activity_vocab', 'resource', 'resource_vocab']:
                desired_df.at[idx, col] = desired_df.loc[idx][col][:desired_idx]

        all_predicted_vocabs = []
        all_predicted_value = []
        for idx in range(len(desired_df)):
            ex_activity = tf.constant([desired_df.iloc[idx]['activity']], dtype=tf.float32)
            ex_resource = tf.constant([desired_df.iloc[idx]['resource']], dtype=tf.float32)
            ex_amount = tf.constant([desired_df.iloc[idx]['amount']], dtype=tf.float32)
            out, _ = self.pred_model(ex_activity, ex_resource, ex_amount, training=False)
            out = tf.nn.softmax(out, axis=-1)
            pred_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]
            predicted_vocab =  self.pred_model.activity_vocab.index_to_vocab(pred_idx)
            all_predicted_vocabs.append(predicted_vocab)
            all_predicted_value.append(out[:, -1, pred_idx].numpy()[0])


        desired_df = pd.DataFrame(desired_df)
        desired_df['Predicted_vocab'] = all_predicted_vocabs
        desired_df['Predicted_value'] = all_predicted_value

        return desired_df

        






        
