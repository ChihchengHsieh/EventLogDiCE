from enum import Enum
from model.PredNextBERT import BERTScheduler

class SelectableDatasets(Enum):

    #############
    # PM
    #############

    BPI2012 = "BPI2012"
    Helpdesk = "Helpdesk"
    BPI2012WithResource = "BPI2012WithResource"

    #############
    # Medical
    #############
    Diabetes = "Diabetes"
    BreastCancer = "BreastCancer"


class SelectableLoss(Enum):
    CrossEntropy = "CrossEntropy"
    HingeLoss = "HingeLoss"
    BCE = "BCE"


class SelectableOptimizer(Enum):
    Adam = "Adam"
    SGD = "SGD"
    RMSprop = "RMSprop"


class SelectableLrScheduler(Enum):
    StepScheduler = "StepScheduler"
    ExponentialDecay= "ExponentialDecay"
    BERTScheduler = "BERTScheduler"
    NotUsing = "NotUsing"

class BPI2012ActivityType(Enum):
    O = "O"
    A = "A"
    W = "W"
