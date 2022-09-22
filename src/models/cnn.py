from src.models.abstract_model import Model
from src.data.dataset import Dataset
from src.utils.dataset_tools import load_data
from tensorflow.keras import layers, models

import tensorflow as tf
import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class CNN(Model):
    def __init__(self, model_path, type='multiclass'):
        super().__init__(model_path, type)
        self.create_model()

    def create_slim_model(self, input=(100,100,3), output=15):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=input))
        self.model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        self.model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
        self.model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        self.model.add(layers.Flatten())
        
        self.model.add(layers.Dense(units=512,activation="relu"))
        self.model.add(layers.Dense(units=output, activation="sigmoid"))

        self.model.compile(optimizer='adam',
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    
    def train_model_in_batch (self, data, labels, batch_size=64, epochs=60):

        trainDS = tf.data.Dataset.from_tensor_slices((data, labels))
        trainDS = (trainDS
            .map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        trainDS = trainDS.map(
            lambda image, label: (tf.image.per_image_standardization(image), label)
        )
        self.model.fit(trainDS, epochs=epochs)


    # overriding abstract method
    def create_model(self, input_shape=(32, 32, 3), num_classes=10, to_save=False):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(num_classes))


        self.model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        if to_save:
            self.save_model()

    # overriding abstract method
    def evaluate_model(self, data, labels):
        return self.model.evaluate(data,  labels, verbose=0)

    def train_model(self, dataset: Dataset, epochs=20):
        (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data()

        self.model.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_test, y_test))
        
        # logger.info('Model trainning done')
        # test_loss, test_acc = self.evaluate_model(x_test, y_test)

        # logger.info('Model accuracy %s', test_acc)
        # logger.info('Model loss: %s', test_loss)

        #self.save_model()