from functools import update_wrapper
from inspect import trace
import numpy as np
import tensorflow as tf
from utils.print import print_block
import time


class EventLogDiCE():
    def __init__(self, activity_vocab, resource_vocab, possible_amount, possible_activities, possible_resources, pred_model, scenario_model):
        '''
        [activity_vocab]: `VocabDict` for activity

        [resource_vocab]: `VocabDict` for resource

        [possible_amount]: List contains `[min_amount, max_amount]`

        [possible_activity]: Availible actiivty to replace current activiity. note: it should
        contain allall the vocabs of activity in your tracethat you want to generate counterfactual.

        [possible_resource]: Availible actiivty to replace current resource. note: it should
         contain all the vocabs of resource in your trace that you want to generate counterfactual.

        [pred_model]: prediction model, it should be wrapped by DiCEDifferentiableWrapper.

        [scenario_model]: the model used for calculating the scenario loss.

        '''
        self.possible_activities = possible_activities
        self.possible_amount = possible_amount
        self.possible_resources = possible_resources
        self.pred_model = pred_model
        self.scenario_model = scenario_model
        self.activity_vocab = activity_vocab
        self.resource_vocab = resource_vocab

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

    def transform_to_ohe_normalized_input(self, activities, resources):
        '''
        Tranform to one-hot encoding format.
        '''
        activity_cf = tf.one_hot(
            activities, depth=len(self.possible_activities))
        resource_cf = tf.one_hot(resources, depth=len(self.possible_resources))
        return activity_cf, resource_cf

    def get_valid_cf(self, amount_cf, ohe_activity_cf, ohe_resource_cf, use_sampling=False):
        '''
        Transform the updated CF to one-hot encoding format.
        '''

        if use_sampling:
            return (tf.clip_by_value(amount_cf, self.possible_amount[0], self.possible_amount[1]),
                    tf.squeeze(tf.one_hot(tf.random.categorical(
                        ohe_activity_cf, 1), depth=len(self.possible_activities)), axis=1),
                    tf.squeeze(tf.one_hot(tf.random.categorical(ohe_resource_cf, 1), depth=len(self.possible_resources)), axis=1))

        return amount_cf, tf.one_hot(tf.argmax(ohe_activity_cf, axis=-1), depth=len(self.possible_activities)), tf.one_hot(tf.argmax(ohe_resource_cf, axis=-1), depth=len(self.possible_resources))

    def get_valid_scenario_loss(self, cf_input, scenario_using_hinge_loss):
        '''
        Get the scenario loss from scenario model.
        '''
        out, _ = self.scenario_model(*cf_input, training=False)
        self.scenario_out = out

        # get loss
        if scenario_using_hinge_loss:
            loss = tf.reduce_sum(
                tf.keras.metrics.hinge(tf.ones_like(out), out))
        else:
            out = tf.nn.sigmoid(out)
            loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(
                y_true=tf.ones_like(out), y_pred=out,  from_logits=False
            ))

        return loss

    def map_to_original_vocabs(self, reduced, original, input, trace_len):
        '''
        Expect one-hot encoded input.
        '''
        after_ = [None] * len(original)
        for i, a in enumerate(reduced):
            dest_index = original.index(a)
            after_[dest_index] = input[:, i:i+1]
        after_ = tf.concat([tf.zeros((trace_len, 1))
                           if a is None else a for a in after_], axis=1)
        return after_

    def transform_to_model_input(self, activitiy_cf, resource_cf, amount_cf, trace_len):
        '''
        Transform the given input to the dimension that our pred model is trained on.
        '''

        activitiy = self.map_to_original_vocabs(
            self.possible_activities,
            self.activity_vocab.vocabs,
            activitiy_cf,
            trace_len,
        )[tf.newaxis, :, :]

        activity_sos_oh = tf.one_hot(self.activity_vocab.sos_idx(), depth=len(
            self.activity_vocab))[tf.newaxis, tf.newaxis, :]

        activitiy = tf.concat(
            [
                activity_sos_oh,
                activitiy
            ],
            axis=1
        )

        resources = self.map_to_original_vocabs(
            self.possible_resources,
            self.resource_vocab.vocabs,
            resource_cf,
            trace_len,
        )[tf.newaxis, :, :]

        resource_sos_oh = tf.one_hot(self.resource_vocab.sos_idx(), depth=len(
            self.resource_vocab))[tf.newaxis, tf.newaxis, :]

        resources = tf.concat(
            [
                resource_sos_oh,
                resources
            ],
            axis=1
        )

        amount = amount_cf
        return [activitiy, resources, amount]

    def get_pred_model_ouptut(self, data_input):
        '''
        Get pred model output.
        '''
        traces, resources, amount = data_input
        out, _ = self.pred_model(traces, resources, amount, training=False)
        predicted_idx = tf.argmax(out[:, -1, :], axis=-1).numpy()[0]
        return out[:, -1, :],  predicted_idx

    def run_pls(self,
                amount_input,
                idx_activities_no_tag,
                idx_resources_no_tag,
                desired_vocab,
                use_valid_cf_only=False,
                use_sampling=True,
                scenario_using_hinge_loss=True,
                class_using_hinge_loss=True,
                use_clipping=True,
                updating_activity_cf=True,
                updating_resource_cf=True,
                updating_amount_cf=True,
                scenario_threshold=0.5,
                scenario_ask_to_continue=True,
                class_loss_weight=1.0,
                distance_loss_weight=1e-8,
                cat_loss_weight=0.0,
                scenario_weight=1.0,
                verbose_freq=20,
                max_iter=200,
                lr=0.05,
                ):
        '''
        This function run the DiCE algorithm trying to find the counterfacutal
        for input query.

        Args:

        [amount_input]: 1-D numpy array contains the value of amount.

        [idx_activities_no_tag]: 1-D index list contains the activities.

        [idx_resources_no_tag]: 1-D index list contains the resources.

        [desired_vocab]: The desired vocab we want to generate the CF for.

        [use_valid_cf_only]: Wehther to replace the current CF by a valid
        (one-hot encoded) for next iteration.

        [use_sampling]: (Work only when `use_valid_cf_only=True`). When it's `True`, we sample from
         current vocabulary distribution, rather than taking the one with max value.

        [scenario_using_hinge_loss]: Whether to use hinge loss for calculating the scenario loss.
        Note: the scenario model in example is trained with binary cross entropy.

        [class_using_hinge_loss]: Whetehr to use hinge loss for calculating the class (CF) loss.
        Note: the pred model in example is trained with cross entropy.

        [use_clipping]: If `True`, clip the amount in `[min_amount, max_amount]`, where you provided
         in the possible amount. Also, clip the categorical CF value to [0, 1]

        [updating_activity_cf]: Whether to calculate the gradient of activity and update it.

        [updating_resource_cf]: Whether to calculate the gradient of resource and udpate it.

        [updating_amount_cf]: Whether to calculate the graident of amount and udpate it.

        [scenario_threshold]: The threshold for output a valid scenario. The scenario model
        will calculate a value for each CF step. This functino will only stop and return CF
        when the scenario model the mean value of each step is > scenario_threshold.

        [scenario_ask_to_continue]: If `scenario_ask_to_continue=True`, when a counterfactual found with
        low scenario score, it will ask the user if you want to continue.

        # When `scenario_threshold=True`, only output when below statement return true.
        >>>
        >>> np.mean(scenarios_values_in_each_step) > scenario_threshold

        [class_loss_weight]: Loss weight for finding counterfactual.

        [distance_loss_weight]: Loss weight for the distance between current counterfactual
        and query data.

        [cat_loss_weight]: Loss weight for categorical data, which retricting the sum value of
        the updated one-hot encoded tensor should be 1.

        [scenario_weight]: Loss weight for scenario loss.

        [verbose_freq]: Print the current state every `verbose_freq` steps.

        [max_iter]: Max iteration to run for finding counterfactual.

        [lr]: learning rate for DiCE.
        '''

        start_at = time.time()

        # Setting up desired.
        desired_vocab_idx = self.activity_vocab.vocab_to_index(desired_vocab)

        self.desired_vocab = desired_vocab
        self.desired_vocab_idx = desired_vocab_idx

        vocab_activity_no_tag = [self.possible_activities[r]
                                 for r in idx_activities_no_tag]
        vocab_resource_no_tag = [self.possible_resources[r]
                                 for r in idx_resources_no_tag]

        ohe_activity_cf, ohe_resource_cf = self.transform_to_ohe_normalized_input(
            idx_activities_no_tag, idx_resources_no_tag)

        trace_len = len(idx_activities_no_tag)  # Need <SOS>

        amount_cf = tf.Variable(amount_input)
        ohe_activity_cf = tf.Variable(ohe_activity_cf)
        ohe_resource_cf = tf.Variable(ohe_resource_cf)

        ohe_resource_backup = ohe_resource_cf.numpy()
        ohe_activity_backup = ohe_activity_cf.numpy()
        amount_backup = amount_cf.numpy()

        self.amount_cf = amount_cf
        self.ohe_activity_cf = ohe_activity_cf
        self.ohe_resource_cf = ohe_resource_cf

        model_input = self.transform_to_model_input(
            ohe_activity_cf,
            ohe_resource_cf,
            amount_cf,
            trace_len,
        )

        self.model_input = model_input

        _, init_predicted_idx = self.get_pred_model_ouptut(
            model_input
        )

        # dpp_style:inverse_dist

        init_predicted_vocab = self.activity_vocab.index_to_vocab(
            init_predicted_idx)

        # This only used for calculating the loss.

        is_matching = (1.0 if init_predicted_idx ==
                       desired_vocab_idx else 0.0)

        desired_pred = 1 - is_matching

        print_block(
            f"Prediction: [{init_predicted_vocab}({init_predicted_idx})] | Desired: [{desired_vocab}({desired_vocab_idx})]", "Model Prediction"
        )

        print_block(f"[{is_matching:.0f}] ==========> [{desired_pred:.0f}]",
                    "Counterfactual Process")

        for i in range(max_iter):

            optim = tf.keras.optimizers.Adam(learning_rate=lr)

            with tf.GradientTape() as tape:
                # Get prediction from cf
                # Maybe we can convert it to the original input here
                self.ohe_activity_cf = ohe_activity_cf
                self.ohe_resource_cf = ohe_resource_cf
                self.amount_cf = amount_cf
                self.trace_le = trace_len

                self.model_input = model_input

                model_input = self.transform_to_model_input(
                    ohe_activity_cf,
                    ohe_resource_cf,
                    amount_cf,
                    trace_len
                )

                cf_output, cf_pred_idx = self.get_pred_model_ouptut(
                    model_input)
                    
                self.cf_output = cf_output

                # Distance loss
                activity_distance_loss = tf.reduce_sum(
                    tf.pow((ohe_activity_cf - ohe_activity_backup), 2))
                resources_distance_loss = tf.reduce_sum(
                    tf.pow(ohe_resource_cf - ohe_resource_backup, 2))
                amount_distance_loss = self.min_max_scale_amount(
                    tf.pow(amount_cf - amount_backup, 2))
                # amount_distance_loss = tf.pow(amount_cf - amount_backup, 2)
                distance_loss = activity_distance_loss + \
                    resources_distance_loss + amount_distance_loss

                # Categorical contraint
                activity_cat_loss = tf.pow(
                    tf.reduce_sum(ohe_activity_cf, axis=1) - 1, 2)
                resource_cat_loss = tf.pow(
                    tf.reduce_sum(ohe_resource_cf, axis=1) - 1, 2)
                cat_loss = tf.reduce_sum(activity_cat_loss + resource_cat_loss)

                # Class loss
                # Trying to use another loss.
                if class_using_hinge_loss:
                    class_loss = tf.keras.metrics.hinge(
                        desired_pred, tf.nn.sigmoid(cf_output[:, desired_vocab_idx:desired_vocab_idx+1]))
                else:
                    class_loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_true=[desired_vocab_idx],
                        y_pred=cf_output[tf.newaxis, :, :]
                    )

                scenario_loss = self.get_valid_scenario_loss(
                    model_input, scenario_using_hinge_loss)

                # Get total loss
                # the category loss need to prevent the updated cf so different from the original one.
                loss = (class_loss_weight * class_loss) + (distance_loss * distance_loss_weight) + \
                    (cat_loss * cat_loss_weight) + \
                    (scenario_loss * scenario_weight)

            if (i != 0) and i % verbose_freq == 0:
                print_block(f"Total [{loss.numpy().flatten()[0]:.2f}] | " +
                            f"Scenario [{scenario_loss.numpy().flatten()[0]:.2f}] | " +
                            f"Class [{class_loss.numpy().flatten()[0]:.2f}] | " +
                            f"Category [{cat_loss.numpy():.2f}] | " +
                            f"Distance [{distance_loss.numpy().flatten()[0]:.2f}]",
                            f"Step {i} Loss")

            updating_vars = []

            if (updating_amount_cf):
                updating_vars.append(amount_cf)

            if (updating_activity_cf):
                updating_vars.append(ohe_activity_cf)

            if (updating_resource_cf):
                updating_vars.append(ohe_resource_cf)

            grad = tape.gradient(
                loss,  updating_vars)

            self.grad = grad
            optim.apply_gradients(
                zip(grad, updating_vars))

            # Clipping the value for next round (This step has a risk to cause the nn can't find the local minimum.)
            if use_clipping:
                amount_cf.assign(tf.clip_by_value(
                    amount_cf, self.possible_amount[0], self.possible_amount[1]))
                ohe_activity_cf.assign(
                    tf.clip_by_value(ohe_activity_cf, 0.0, 1.0))
                ohe_resource_cf.assign(
                    tf.clip_by_value(ohe_resource_cf, 0.0, 1.0))

            # Get a valid cf close to current cf.
            temp_amount_cf, temp_ohe_activity_cf, temp_ohe_resource_cf = self.get_valid_cf(
                amount_cf, ohe_activity_cf, ohe_resource_cf)

            temp_model_input = self.transform_to_model_input(
                temp_ohe_activity_cf, temp_ohe_resource_cf, temp_amount_cf, trace_len)  # Updated.

            self.temp_model_input = temp_model_input

            # Get prediction from the valid cf.
            _, temp_predicted_idx = self.get_pred_model_ouptut(
                temp_model_input)

            if (i != 0) and i % verbose_freq == 0:
                cf_pred_vocab = self.activity_vocab.index_to_vocab(
                    cf_pred_idx)
                # print_block(f"{cf_pred_vocab} ({cf_pred_idx})",
                #           "Invalid CF predicted")
                temp_pred_vocab = self.activity_vocab.index_to_vocab(
                    temp_predicted_idx)
                print_block(f"Invalid: {cf_pred_vocab} ({cf_pred_idx}) | Valid: {temp_pred_vocab} ({temp_predicted_idx})",
                            f"Step {i} CF predicted")

            if (use_valid_cf_only):
                # Replace current cf by valid cf.
                amount_cf.assign(temp_amount_cf)
                if (use_sampling):
                    # Using sampling
                    sample_amount, sample_activity, sample_resource = self.get_valid_cf(
                        amount_cf, ohe_activity_cf, ohe_resource_cf, use_sampling=True)
                    amount_cf.assign(sample_amount)
                    ohe_activity_cf.assign(sample_activity)
                    ohe_resource_cf.assign(sample_resource)
                else:
                    # Using argmax
                    ohe_activity_cf.assign(temp_ohe_activity_cf)
                    ohe_resource_cf.assign(temp_ohe_resource_cf)

            if (temp_predicted_idx == desired_vocab_idx):

                temp_scenario = tf.nn.sigmoid(self.scenario_model(
                    *temp_model_input, training=False)[0]).numpy()

                valid_scanrio = np.mean(temp_scenario) > scenario_threshold

                if valid_scanrio:
                    print_block(f"Running time: {time.time() - start_at:.2f}",
                                f"! Counterfactual found in step [{i+1}] \U0001F917 !")
                else:
                    print_block(f"Running time: {time.time() - start_at:.2f}",
                                f"! Counterfactual with low scenario score found in step [{i+1}] \U0001F648 !")

                activity_out = [self.possible_activities[i]
                                for i in tf.argmax(temp_ohe_activity_cf, axis=-1).numpy()]
                resource_out = [self.possible_resources[i]
                                for i in tf.argmax(temp_ohe_resource_cf, axis=-1).numpy()]
                amount_out = amount_cf.numpy()[0]

                print_block(amount_input, "Input Amount")
                print_block(vocab_activity_no_tag, "Input Activities")
                print_block(vocab_resource_no_tag, "Input Resource")

                print_block(amount_out, "Valid CF Amount")
                print_block(activity_out, "Valid CF Activities")
                print_block(resource_out, "Valid CF Resource")

                print_block(np.around(temp_scenario.flatten(),
                            decimals=1), "Valid CF scenario output")

                if (valid_scanrio):
                    return amount_out, activity_out, resource_out
                else:
                    if (scenario_ask_to_continue):
                        asnwer = input("Do you still want to continue")
                        if asnwer.lower().startswith("y"):
                            continue
                        else:
                            return amount_out, activity_out, resource_out

        activity_out = [self.possible_activities[i]
                        for i in tf.argmax(temp_ohe_activity_cf, axis=-1).numpy()]
        resource_out = [self.possible_resources[i]
                        for i in tf.argmax(temp_ohe_resource_cf, axis=-1).numpy()]
        amount_out = amount_cf.numpy()[0]

        print_block("No valid counterfactual found! \U0001F630")

        print_block(amount_input, "Input Amount")
        print_block(vocab_activity_no_tag, "Input Activities")
        print_block(vocab_resource_no_tag, "Input Resource")

        print_block(amount_out, "Current CF Amount")
        print_block(activity_out, "Current CF Activities")
        print_block(resource_out, "Current CF Resource")

        return amount_out, activity_out, resource_out
