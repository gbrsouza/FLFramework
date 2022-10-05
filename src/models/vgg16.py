from src.models.abstract_model import Model
from src.data.dataset import Dataset
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16

import tensorflow as tf
import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class LocalVGG16(Model):
    def __init__(self, model_path, type='multiclass'):
        super().__init__(model_path, type)
        self.create_model()

    # overriding abstract method
    def create_model(self, input_shape=(32,32,3), num_classes=10, to_save=False):
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = False
        
        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(50, activation='relu')
        dense_layer_2 = layers.Dense(20, activation='relu')
        prediction_layer = layers.Dense(num_classes, activation='softmax')

        self.model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer
        ])

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy'],
        )

        if to_save:
            self.save_model()

    def train_model(self, dataset: Dataset, epochs=10):
        (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data()

        self.model.fit(x_train, y_train, epochs=epochs, 
                    validation_data=(x_test, y_test))
        
        logger.info('Model trainning done')
        test_loss, test_acc = self.model.evaluate(x_test,  y_test, verbose=0)

        logger.info('Model accuracy %s', test_acc)
        logger.info('Model loss: %s', test_loss)

        #self.save_model()