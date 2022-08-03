from src.data.dataset import Dataset
from src.models.abstract_model import Model
from abc import abstractmethod

class FLAlgorithm():
    def __init__(self) -> None:
        pass

    @abstractmethod
    def improve_model (self, sample: Dataset, model: Model, lr=0.1): 
        """This function impplements the local improvement step

        Args:
            sample (Dataset): The local dataset from client
            model (Model): The model to improve
            lr (float): The learning rate to apply in the optimization
        """
        pass