from src.models.abstract_model import Model
from src.data.dataset import Dataset
from src.federated_learning.abstract_fl_algorithm import FLAlgorithm
from src.utils.evaluator import evaluate_model
from sklearn.metrics import accuracy_score
from src.utils.dataset_tools import processing_image_dataset

import logging as log
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.get_logger().setLevel('ERROR')
log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class Client():

    def __init__(self, local_model: Model, local_dataset: Dataset, improver: FLAlgorithm):
        self.local_model = local_model
        self.local_dataset = local_dataset
        self.improver = improver

    def run(self, epochs, batch_size=8):
        """run the local improvement step

        Args:
            epochs (int): number of epochs
        """
        
        for epoch in range(epochs):
            logger.info("Starting epoch %s of %s", epoch, epochs)
            start_time = time.time()
            (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = self.local_dataset.split_data()
            data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            data = data.shuffle(buffer_size=1024).batch(batch_size)
  
            # Iterate over the batches of the dataset.
            actual_model = self.local_model.get_model()
            for (x_batch_train, y_batch_train) in data:
                #x_batch_train, y_batch_train = processing_image_dataset(x_batch_train, y_batch_train, (100,100))
                actual_model, loss_value = self.improver.improve_model(x_batch_train, y_batch_train, actual_model)
  
            # update local model
            self.local_model.set_model(actual_model)
            end_time = time.time() 

            # evaluating model 
            logger.info("Evaluating local model")
            #x_test, y_test = processing_image_dataset(x_test, y_test, (100,100))
            acc, pre, rec, confu_matrix = evaluate_model(x_test, y_test, self.local_model)

            logger.info('\n-------confusion matrix-------\n%s', confu_matrix)
            logger.info('metrics after epoch %s', epoch)
            logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)
            logger.info('Time taken: %.2fs', end_time - start_time)

    def get_local_model(self):
        return self.local_model