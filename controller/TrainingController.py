from tensorflow.python.keras.utils import losses_utils
from parameters.training import LossParameters, OptimizerParameters, TrainingParameters
from utils.preprocessing import dataset_split
from typing import List, Tuple

import pandas as pd
from utils.exceptions import NotSupportedError
from parameters.enum import SelectableLoss, SelectableLrScheduler, SelectableOptimizer
from utils.print import print_block, print_peforming_task
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import os
from utils.save import get_json_dict, load_parameters, save_parameters_json_dict


class TrainingController(object):
    parameters_file_name = 'parameters.json'

    #########################################
    #   Initialisation
    #########################################

    def __init__(self,
                 dataset,
                 model,
                 parameters: TrainingParameters,
                 optim_params: OptimizerParameters,
                 loss_params: LossParameters,
                 ):

        # store parameters
        self.parameters: TrainingParameters = parameters
        self.optim_params = optim_params
        self.loss_params = loss_params

        temp = tf.constant([0])
        print_block("Running on %s " % (temp.device))
        del temp

        ############ Initialise counters ############
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.test_accuracy: float = None
        self.dataset = dataset
        self.model = model

        self.__initialise_dataset()
        self.__intialise_optimizer()
        self.__initialise_loss_fn()

    def __initialise_dataset(self):

        # Initialise the index for splitting the dataset.
        self.train_dataset, self.test_dataset, self.validation_dataset = dataset_split(
            list(range(len(self.dataset))),
            self.parameters.train_test_split_portion,
            seed=self.parameters.random_seed,
            shuffle=True
        )

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            self.train_dataset).batch(self.parameters.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            self.test_dataset).batch(self.parameters.batch_size)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            self.validation_dataset).batch(self.parameters.batch_size)

    def __intialise_optimizer(
        self,
    ):
        # Initialise schedualer
        if self.optim_params.lr_scheduler == SelectableLrScheduler.NotUsing:
            learning_rate = self.optim_params.learning_rate
        elif self.optim_params.lr_scheduler == SelectableLrScheduler.ExponentialDecay:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.optim_params.learning_rate,
                decay_steps=self.optim_params.lr_exp_decay_scheduler_step,
                decay_rate=self.optim_params.lr_exp_decay_scheduler_rate,
                staircase=self.optim_params.lr_exp_decay_scheduler_staircase,
            )
        else:
            raise NotSupportedError(
                "Lr scheduler you selected is not supported")

        # Setting up optimizer
        if self.optim_params.optimizer == SelectableOptimizer.Adam:
            self.optim = tf.keras.optimizers.Adam(learning_rate)
        elif self.optim_params.optimizer == SelectableOptimizer.SGD:
            self.optim = tf.keras.optimizers.SGD(learning_rate)
        elif self.optim_params.optimizer == SelectableOptimizer.RMSprop:
            self.optim = tf.keras.optimizers.RMSprop(learning_rate)
        else:
            raise NotSupportedError("Optimizer you selected is not supported")

    def __initialise_loss_fn(self):
        # Setting up loss
        if self.loss_params.loss == SelectableLoss.CrossEntropy:
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=losses_utils.ReductionV2.NONE
            )
        elif self.loss_params.loss == SelectableLoss.BCE:
            self.loss = tf.keras.losses.BinaryCrossentropy(
                reduction=losses_utils.ReductionV2.NONE
            )
        elif self.loss_params.loss == SelectableLoss.HingeLoss:
            self.loss = tf.keras.metrics.hinge
        else:
            raise NotSupportedError(
                "Loss function you selected is not supported")

    ##########################################
    #   Train & Evaluation
    ##########################################

    def train_step(
        self, data
    ) -> Tuple[float, float]:
        with tf.GradientTape() as tape:
            out, loss, accuracy = self.step(data, training=True)
        self.loss_v = loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.grads = grads
        self.optim.apply_gradients(grads_and_vars=zip(
            grads, self.model.trainable_variables))
        return out, loss.numpy(), accuracy

    def eval_step(
        self, data
    ):
        out, loss, accuracy = self.step(data, training=False)
        return out, loss.numpy(), accuracy

    def step(self, data, training=None):
        out = self.model.data_call(data, training=training)
        loss = self.model.get_loss(self.loss, out, data)
        accuracy = self.model.get_accuracy(out, data)
        return out, loss, accuracy

    def train(
        self,
    ):
        train_summary_writer, test_summary_writer = self.prepare_tensorboard()
        self.train_start_print()

        while self.not_end():
            print_block("Start epoch %d" % (self.current_epoch))

            for _, train_idxs in enumerate(
                self.train_dataset
            ):
                # Retrieve data from training set
                train_data = self.dataset.collate_fn(train_idxs)

                # Train for one step
                _, train_loss, train_accuracy = self.train_step(
                    train_data
                )

                self.current_step += 1

                self.write_accuracy_loss_record(
                    train_summary_writer, train_accuracy, train_loss, self.current_step)

                if self.current_step > 0 and self.current_step % self.parameters.run_validation_freq == 0:
                    (
                        validation_loss,
                        validation_accuracy,
                    ) = self.perform_eval_on_dataset(self.validation_dataset, show_report=False)

                    self.write_accuracy_loss_record(
                        test_summary_writer, validation_accuracy, validation_loss, self.current_step)

            self.current_epoch += 1

        return self.perform_eval_on_testset()

    def model_info(self,):
        self.model(**self.model.get_example_input())
        self.model.summary()

    def perform_eval_on_dataset(self, dataset, show_report: bool = False) -> Tuple[float, float]:

        all_loss = []
        all_accuracy = []
        all_batch_size = []
        all_predictions = []
        all_targets = []

        for idxs in dataset:
            data = self.dataset.collate_fn(idxs)
            out, loss, accuracy = self.eval_step(data)
            all_predictions.extend(
                self.model.get_prediction_list_from_out(out, data))
            all_targets.extend(self.model.get_target_list_from_target(data))
            all_loss.append(loss)
            all_accuracy.append(accuracy)
            all_batch_size.append(len(data[0]))
        self.all_accuracy = all_accuracy
        self.all_target = all_targets
        self.all_predictions = all_predictions
        self.out = out

        accuracy = accuracy_score(all_targets, all_predictions)
        self.accuracy = accuracy
        mean_loss = sum(tf.constant(all_loss) * tf.constant(all_batch_size,
                        dtype=tf.float32)) / len(list(dataset.unbatch().as_numpy_iterator()))

        print_block(
            "Evaluation result | Loss [%.4f] | Accuracy [%.4f] "
            % (mean_loss.numpy(), accuracy)
        )

        if (show_report):
            print_block("Classification Report")
            report = classification_report(all_targets, all_predictions, zero_division=0, output_dict=True, labels=list(
                range(len(self.model.get_labels()))), target_names=list(self.model.get_labels()))
            print(pd.DataFrame(report))

        return mean_loss.numpy(), accuracy

    def perform_eval_on_testset(self):
        print_peforming_task("Testing")
        _, test_accuracy = self.perform_eval_on_dataset(
            self.test_dataset, show_report=False)
        return test_accuracy

    def prepare_tensorboard(self,):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_folder_name = 'logs/gradient_tape/' + current_time
        train_log_dir = tb_folder_name + '/train'
        test_log_dir = tb_folder_name + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        tf.keras.callbacks.TensorBoard(log_dir=train_log_dir)
        print_block("Training records in %s" % (tb_folder_name))
        return train_summary_writer, test_summary_writer

    def train_start_print(self):
        print_block("Total epochs: %d" % (self.parameters.stop_epoch))
        print_block("Total steps: %d" %
                  (self.parameters.stop_epoch * len(self.train_dataset)))

    def not_end(self,):
        return self.current_epoch < self.parameters.stop_epoch

    def write_accuracy_loss_record(self, writer, accuracy, loss, step):
        with writer.as_default():
            tf.summary.scalar('accuracy', accuracy, step=step)
            tf.summary.scalar('loss', loss, step=step)

    def save_parameters(self, folder_path):
        # Save all parameters in the single file

        all_parameters = {}
        all_parameters['train'] = get_json_dict(self.parameters)
        all_parameters['model'] = get_json_dict(self.model.parameters)
        all_parameters['dataset'] = get_json_dict(self.dataset.parameters)
        all_parameters['loss'] = get_json_dict(self.loss_params)
        all_parameters['optim'] = get_json_dict(self.optim_params)

        os.makedirs(folder_path, exist_ok=True)

        # Save parameters
        parameters_saving_path = os.path.join(
            folder_path, TrainingController.parameters_file_name
        )

        save_parameters_json_dict(parameters_saving_path, all_parameters)

    def load_parameters(self, folder_path):

        parameters_json = load_parameters(
            folder_path, TrainingController.parameters_file_name)

        parameters_json['train'] = TrainingParameters(
            **(parameters_json['train']))
        parameters_json['loss'] = LossParameters(**(parameters_json['loss']))
        parameters_json['optim'] = OptimizerParameters(
            **(parameters_json['optim']))

        return parameters_json
