from abc import abstractmethod
from typing import List

class Reader():

    def __init__(self, source):
        """constructor

        Args:
            source (String): The dataset path
        """
        self.source = source

    @abstractmethod
    def read_dataset(self):
        """Function to read a specific dataset"""
        pass