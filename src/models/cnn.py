from src.models.abstract_model import Model
from src.data.dataset import Dataset
from tensorflow.keras import layers, models

import tensorflow as tf
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class CNN(Model):
    def __init__(self, model_path):
        super().__init__(model_path)

    # overriding abstract method
    def create_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))

        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        self.save_model()

    # overriding abstract method
    def evaluate_model(self, data, labels):
        return self.model.evaluate(data,  labels, verbose=0)

    def train_model(self, dataset: Dataset):
        (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data()

        self.model.fit(x_train, y_train, epochs=20, 
                    validation_data=(x_test, y_test))
        
        logger.info('Model trainning done')
        test_loss, test_acc = self.evaluate_model(x_test, y_test)

        logger.info('Model accuracy %s', test_acc)
        logger.info('Model loss: %s', test_loss)

        self.save_model()