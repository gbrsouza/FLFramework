from src.models.abstract_model import Model
from src.data.dataset import Dataset
from src.federated_learning.abstract_fl_algorithm import FLAlgorithm

import logging as log
import time
import tensorflow as tf
from tensorflow import keras

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class Client():

    def __init__(self, local_model: Model, local_dataset: Dataset, improver: FLAlgorithm):
        self.local_model = local_model
        self.local_dataset = local_dataset
        self.improver = improver

    def run(self, epochs, batch_size=64):
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
            for step, (x_batch_train, y_batch_train) in enumerate(data):
                sample = Dataset(x_batch_train, y_batch_train)
                actual_model, loss_value = self.improver.improve_model(sample, actual_model)

                # Log every 20 batches.
                if step % 20 == 0:
                    logger.info(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    logger.info("Seen so far: %d samples" % ((step + 1) * batch_size))
            
            # update local model
            self.local_model.set_model(actual_model)
            end_time = time.time() 

            loss, acc = self.local_model.evaluate_model(x_test, y_test)
            logger.info('metrics after epoch %s', epoch)
            logger.info('loss value: %d, accuracy: %d', loss, acc)
            logger.info('Time taken: %.2fs', end_time - start_time)

    def get_local_model(self):
        return self.local_model