import tensorflow as tf

class ControllerModel(tf.keras.Model):

    def __init__(self) -> None:
        super().__init__()

    def data_call(self, data, training=None):
        pass

    def get_accuracy(self, y_pred, data):
        pass

    def get_loss(self, loss_fn: callable, y_pred, data):
        pass
