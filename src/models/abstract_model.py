from abc import abstractmethod

import os
import logging as log

from src.data.dataset import Dataset

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class Model():

    def __init__(self, model_path, type='binary'):
        self.model_path = model_path
        self.type = type
        self.model = None

    @abstractmethod
    def create_model(self):
        """ create the model structure """
        pass

    def save_model(self):
        """ save the actual model in the model path """
        if self.model != None:
            self.model.save_weights(self.model_path)
            logger.info('\t Model saved successfully at %s', self.model_path)
        else:
            logger.warning('\t No model to be saved')

    @abstractmethod
    def evaluate_model(self, data, labels):
        """evaluate the model 

        Args:
            data (list): A list of data to test
            labels (list): A list of labels 
        """
        pass

    @abstractmethod
    def train_model(self, dataset: Dataset):
        """train the model with a received dataset

        Args:
            data (list): a list of data to train the dataset
            labels (list): a list of labels
        """
        pass

    def load_or_create_model(self, dataset: Dataset):
        """load a previous saved model, if this model not exists, train a new
        model with the received dataset 

        Args:
            data (list): a list of data to train the model
            labels (list): a list of labels
        """
        if os.path.exists(self.model_path):
            logger.info("A pre-trained model was found, loading...")
            self.create_model()
            self.model.load_weights(self.model_path)
            return self.model
        else:
            logger.info("No pre-trained model was found, training a new model")
            self.create_model()
            self.train_model(dataset)
            self.save_model()
            return self.model

    @abstractmethod
    def predict(self, x_test):
        """predict class from model

        Args:
            x_test (Array): The test dataset
        """
        pass

    def get_weights(self):
        """get all weiths from the actual model

        Returns:
            a list of weigths
        """
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model

    def get_type(self):
        return self.type