from src.models.abstract_model import Model
from keras.models import Sequential
from src.data.dataset import Dataset
from src.utils.dataset_tools import load_data
from tensorflow.keras import layers, models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from src.utils.dataset_tools import load_data
from tensorflow.keras.optimizers import RMSprop, SGD

import tensorflow as tf
import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class Inception(Model):
    def __init__(self, model_path, type='multiclass'):
        super().__init__(model_path, type)
        self.create_model()

    # overriding abstract method
    def create_model(self, input_shape=(100,100,3), num_classes=2, to_save=False):
        base_model  = InceptionV3(
            input_shape = input_shape,
            include_top = False,
            weights = 'imagenet'
        )
        base_model.trainable = False

        add_model = Sequential()
        add_model.add(base_model)
        add_model.add(layers.GlobalAveragePooling2D())
        add_model.add(layers.Dropout(0.4))
        add_model.add(layers.Dense(num_classes, activation='softmax'))

        self.model = add_model

        self.model.compile(optimizer = 'adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=['accuracy'])

        if to_save:
            self.save_model()

    # overriding abstract method
    def evaluate_model(self, data, labels):
        return self.model.evaluate(data,  labels, verbose=0)

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