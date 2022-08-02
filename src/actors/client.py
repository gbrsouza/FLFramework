from src.models.abstract_model import Model
from src.data.dataset import Dataset

class Client():

    def __init__(self, global_model: Model, local_dataset: Dataset):
        self.gloval_model = global_model
        self.local_dataset = local_dataset

    