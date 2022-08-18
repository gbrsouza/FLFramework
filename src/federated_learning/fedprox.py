from src.federated_learning.abstract_fl_algorithm import FLAlgorithm
from copy import deepcopy

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class FedProx(FLAlgorithm):
    def __init__(self, global_model) -> None:
        super().__init__()
        self.global_model = global_model

    def difference_models_norm_2(self, model_1, model_2):
        """Calculates the squared l2 norm of a model difference (i.e.
        local_model - global_model)
        Args:
            model_1: the original model
            model_2: the current, in-training model
        Returns: the squared norm
        """


        total_norm = 0.0
        for i in range(len(model_1)):
            if model_1[i].ndim > 3:
                norm = tf.norm(tf.norm((tf.math.subtract(model_1[i],model_2[i])),ord='euclidean', axis=[-2,-1]), ord='euclidean', axis=[-2,-1])
            else: 
                norm = tf.norm((tf.math.subtract(model_1[i],model_2[i])),ord='euclidean', axis=None)
            total_norm += norm
       
        return total_norm

    # overriding abstract method
    def improve_model (self, tensor_data, tensor_labels, model, lr=0.1, mu=0.01): 

        # Instantiate an optimizer and loss function.
        optimizer = keras.optimizers.SGD(learning_rate=lr)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        with tf.GradientTape() as tape:
            logits = model(tensor_data, training=True)
            loss_value = loss_fn(tensor_labels, logits)
            loss_value += mu/2.0 * self.difference_models_norm_2(model.get_weights(), self.global_model.get_weights())
        
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        return model, loss_value
