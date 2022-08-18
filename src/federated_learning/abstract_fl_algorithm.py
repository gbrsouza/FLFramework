from src.data.dataset import Dataset
from src.models.abstract_model import Model
from abc import abstractmethod

class FLAlgorithm():
    def __init__(self) -> None:
        pass

    @abstractmethod
    def improve_model (self, tensor_data, tensor_labels, model, lr=0.1, mu=0.0): 
        """This function impplements the local improvement step

        Args:
            tensor_data (EagerTensor): The batch data from dataset to improve model
            tensor_labels(EagerTensor): The labels from the tensor data
            model (Model): The model to improve
            lr (float): The learning rate to apply in the optimization
        """
        pass