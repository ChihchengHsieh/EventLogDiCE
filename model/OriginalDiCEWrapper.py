import tensorflow as tf
from tensorflow.python.keras.backend import dtype


class OriginalDiCEWrapper(tf.keras.Model):
    '''
    It's a new model classifying where the destination is prefered.
    '''

    def __init__(self, model, activity_vocab, resource_vocab, desired: int, trace_length: int, possible_activities, possible_resources, possible_amount):
        super(OriginalDiCEWrapper, self).__init__()
        self.model = model
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab
        self.desired = desired
        self.trace_length = trace_length
        self.possible_activities = possible_activities
        self.possible_resources = possible_resources
        self.possilbe_amount = possible_amount

        self.all_predicted = []
        self.all_trace = []
        self.all_model_out = []
        self.all_cf_input = []
        self.all_resource = []
        self.all_amount = []


    def call(self, input):
        '''
        Input will be one-hot encoded tensor.
        '''
        # print("Detect input with shape: %s" % str(input.shape))
        self.all_cf_input.append(input.numpy())

        # split_portion = [1, len(self.without_tags_vocabs) * self.trace_length,
        #                 len(self.without_tags_resources) * self.trace_length]

        # self.split_portion = split_portion
        # self.input_data = input

        # amount, traces, resources = tf.split(input, split_portion, axis=1)

        # # print("Origin Amount")
        # # print(amount)
        # amount = (amount * (self.amount_max - self.amount_min)) + self.amount_min
        # # print('Amount scale back')
        # # print(amount)

        # # print("Amount value is %.2f" % (amount.numpy()) )


        ## ! implement a weight propagation model

        # traces = tf.argmax(
        #     tf.stack(tf.split(traces, self.trace_length, axis=-1,), axis=1), axis=-1)

        # resources = tf.argmax(
        #     tf.stack(tf.split(resources, self.trace_length, axis=-1,), axis=1), axis=-1)

        # # transfer to the input with tags.
        # traces = tf.constant(self.vocab.list_of_vocab_to_index_2d(
        #     [[self.without_tags_vocabs[idx] for idx in tf.squeeze(traces).numpy()]]), dtype=tf.int64)

        # resources = tf.constant(
        #     [[self.resources.index(self.without_tags_resources[idx]) for idx in tf.squeeze(resources).numpy()]], dtype=tf.int64)

        # traces =  tf.constant(self.vocab.list_of_vocab_to_index(list(inversed_data[self.activity_feature_names].iloc[0])), dtype= tf.int64)[tf.newaxis ,:]
        # resources =  tf.constant([self.resources.index(r) for r in (list(inversed_data[self.resource_feature_names].iloc[0]))], dtype= tf.int64)[tf.newaxis ,:]
        # amount = tf.constant(inversed_data['amount'], dtype=tf.float32)[tf.newaxis, :]

        amount, traces, resources = self.ohe_to_model_input(input)

        self.all_trace.append(traces.numpy())
        self.all_resource.append(resources.numpy())
        self.all_amount.append(amount.numpy())

        # traces = self.map_to_original_vocabs(self.possible_activities, self.vocab.vocabs, traces)
        # resources = self.map_to_original_vocabs(self.possible_resources, self.resources, resources)

        # # Concate the <SOS> tag in the first step.
        # traces = tf.concat(
        #     [tf.constant([[self.sos_idx_activity]], dtype=tf.int64),  traces], axis=-1)

        # resources = tf.concat(
        #     [tf.constant([[self.sos_idx_resource]], dtype=tf.int64), resources], axis=-1)

        # Feed to the model
        # print("Ready for input")
        out, _ = self.model(traces, resources, tf.squeeze(amount, axis=-1))

        self.all_model_out.append(out.numpy())
        self.all_predicted.append(tf.argmax(out[:, -1, :], axis=-1).numpy())

        return out[:, -1, self.desired: self.desired+1]

    def ohe_to_model_input(self, input):
        amount, activities, resources = tf.split(input, [1, self.trace_length * len(self.possible_activities), self.trace_length * len(self.possible_resources)], axis=1)
        activities =  tf.reshape(activities, [self.trace_length, len(self.possible_activities)])
        activities = self.map_to_original_vocabs(self.possible_activities, self.activity_vocab.vocabs, activities)
        activities = tf.concat([tf.one_hot([self.activity_vocab.sos_idx()], depth= len(self.activity_vocab)),activities], axis = 0)[tf.newaxis, :, :]

        resources = tf.reshape(resources, [self.trace_length, len(self.possible_resources)])
        resources = self.map_to_original_vocabs(self.possible_resources, self.resource_vocab.vocabs, resources)
        resources = tf.concat([tf.one_hot([self.resource_vocab.sos_idx()], depth= len(self.resource_vocab)), resources], axis = 0)[tf.newaxis, :, :]
        amount = (amount * (self.possilbe_amount[1] - self.possilbe_amount[0])) + self.possilbe_amount[0]
        return amount, activities, resources

    def map_to_original_vocabs(self, reduced, original, input):
        after_ = [None] * len(original)
        for i, a in enumerate(reduced):
            dest_index = original.index(a)
            after_[dest_index] = input[:, i:i+1]
        after_ = tf.concat([ tf.zeros(( self.trace_length, 1))  if a is None  else a  for a in after_], axis=1)
        return after_