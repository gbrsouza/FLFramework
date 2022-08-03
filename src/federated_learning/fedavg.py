from src.federated_learning.abstract_fl_algorithm import FLAlgorithm
from src.data.dataset import Dataset
from src.models.abstract_model import Model

import tensorflow as tf
from tensorflow import keras
import time
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class FedAvg(FLAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    # overriding abstract method
    def improve_model (self, sample: Dataset, model: Model, lr=0.1): 
        
        # Instantiate an optimizer and loss function.
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            logits = model(sample.get_data(), training=True)
            loss_value = loss_fn(sample.get_labels(), logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return model, loss_value