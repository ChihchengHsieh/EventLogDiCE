from dataclasses import dataclass


@dataclass
class LSTMScenarioCfModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1


@dataclass
class LSTMPredNextModelParameters(object):
    activity_embedding_dim: int = 32
    resource_embedding_dim: int = 128
    lstm_hidden: int = 64
    dense_dim: int = 64
    dropout: float = 0.1


@dataclass
class PredNextBERTParameters(object):
    num_layers: int = 4
    model_dim: int = 32
    feed_forward_dim: int = 64
    num_heads: int = 4
    dropout_rate: float = .1
