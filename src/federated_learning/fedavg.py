from src.federated_learning.abstract_fl_algorithm import FLAlgorithm

import tensorflow as tf
from tensorflow import keras
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class FedAvg(FLAlgorithm):
    def __init__(self) -> None:
        super().__init__()

    # overriding abstract method
    def improve_model (self, tensor_data, tensor_labels, model, lr=0.01, mu=0.0): 

        # Instantiate an optimizer and loss function.
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            logits = model(tensor_data, training=True)
            loss_value = loss_fn(tensor_labels, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return model, loss_value