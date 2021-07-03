from abc import ABC

class ControllerModel(ABC):

    def __init__(self) -> None:
        super().__init__()

    def data_call(self, data, training=None):
        pass

    def get_folder_path(self, current_file, test_accuracy, additional=""):
        pass

    def get_accuracy(self, y_pred, data):
        pass

    def get_loss(self, loss_fn: callable, y_pred, data):
        pass

    def save(self, folder_path: str):
        pass

    def save_model(self, folder_path: str):
        pass

    def save_parameters(self, folder_path):
       pass

    def load_model(self, folder_path: str):
        pass

    @staticmethod
    def load_model_params(folder_path):
        pass

    @staticmethod
    def load(folder_path):
        pass

    def get_example_input(self,):
        pass

    def get_prediction_list_from_out(self, y_pred, data):
        pass
    
    def get_target_list_from_target(self, data):
        pass
