from typing import List
from dataclasses import dataclass, field
from parameters.enum import (
    SelectableLoss,
    SelectableLrScheduler,
    SelectableOptimizer,
)

@dataclass
class OptimizerParameters(object):
    """
    It will be override once you have load_model and load_optimizer = True
    """

    optimizer: SelectableOptimizer = SelectableOptimizer.Adam

    learning_rate: float = 0.005
    l2: float = 1e-10

    # Scheduler
    lr_scheduler: SelectableLrScheduler = SelectableLrScheduler.ExponentialDecay

    lr_step_scheduler_step: int = 800
    lr_setp_scheduler_gamma: float = .8

    # decay scheduler
    lr_exp_decay_scheduler_step: int = 1000
    lr_exp_decay_scheduler_rate: float = .96
    lr_exp_decay_scheduler_staircase: bool = True

    # SGD
    SGD_momentum: float = 0.9

    def __post_init__(self):
        if (type(self.lr_scheduler) == str):
            self.lr_scheduler = SelectableLrScheduler[self.lr_scheduler]

        if (type(self.optimizer) == str):
            self.optimizer = SelectableOptimizer[self.optimizer]


@dataclass
class LossParameters(object):
    loss: SelectableLoss = SelectableLoss.CrossEntropy


@dataclass
class TrainingParameters(object):
    stop_epoch: int = 10
    batch_size: int = 128
    verbose_freq: int = 250
    run_validation_freq: int = 80
    train_test_split_portion: List[float] = field(
        default_factory=lambda: [0.8, 0.1]
    )
    random_seed: int = 12345
