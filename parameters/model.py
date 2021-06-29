from dataclasses import dataclass


@dataclass
class LSTMPredNextWithResourceModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1


@dataclass
class LSTMScenarioCfWithResourceModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1


@dataclass
class LSTMPredNextAmountSpecificModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1