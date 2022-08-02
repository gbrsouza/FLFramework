from src.data.dataset import Dataset
from src.models.abstract_model import Model
from abc import abstractmethod

class FLAlgorithm():
    def __init__(self) -> None:
        pass

    @abstractmethod
    def improve_local_model (self, dataset: Dataset, model: Model):
        """This function impplements the local improvement step

        Args:
            dataset (Dataset): The local dataset from client
        """
        pass