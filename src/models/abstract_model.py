import logging
import os

from abc import abstractmethod

class Model():

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    @abstractmethod
    def create_model(self):
        """ create the model structure """
        pass

    def save_model(self):
        """ save the actual model in the model path """
        if self.model != None:
            self.model.save_weights(self.model_path)
            logging.info('\t Model saved successfully at %s', self.model_path)
        else:
            logging.warning('\t No model to be saved')

    @abstractmethod
    def evaluate_model(self, data, labels):
        """evaluate the model 

        Args:
            data (list): A list of data to test
            labels (list): A list of labels 
        """
        pass

    @abstractmethod
    def train_model(self, data, labels):
        """train the model with a received dataset

        Args:
            data (list): a list of data to train the dataset
            labels (list): a list of labels
        """
        pass

    def load_or_create_model(self, data: list, labels: list):
        """load a previous saved model, if this model not exists, train a new
        model with the received dataset 

        Args:
            data (list): a list of data to train the model
            labels (list): a list of labels
        """
        if os.path.exists(self.model_path):
            logging.info("A pre-trained model was found, loading...")
            self.create_model()
            self.model.load_weights(self.model_path)
            return self.model
        else:
            logging.info("No pre-trained model was found, training a new model")
            self.create_model()
            self.train_model(data, labels)
            self.save_model()
            return self.model

    def get_weights(self):
        """get all weiths from the actual model

        Returns:
            a list of weigths
        """
        return self.model.get_weights()

    def get_model(self):
        return self.model
    
    def set_model(self, model):
        self.model = model