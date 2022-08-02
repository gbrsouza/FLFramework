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
    def improve_local_model (self, dataset: Dataset, model: Model):
        
        # Instantiate an optimizer.
        optimizer = keras.optimizers.SGD(learning_rate=0.1)

        # Instantiate a loss function.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the metrics.
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        # prepare the training dataset
        logger.info('preparing dataset')
        batch_size = 64

        # reserve 10% samples for validation
        val_size = int(len(self.images)*0.3)*-1
        x_val = self.images[val_size:]
        y_val = self.labels[val_size:]

        x_train = self.images[:val_size]
        y_train = self.labels[:val_size]

        logger.info('x_val size %s', len(x_val))
        logger.info('x_train size %s', len(x_train))

        # prepare the training dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        # Prepare the validation dataset.
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(batch_size)

        @tf.function
        def train_step(self, x, y):
            with tf.GradientTape() as tape:
                logits = self.model(x, training=True)
                loss_value = loss_fn(y, logits)
            grads = tape.gradient(loss_value, self.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
            train_acc_metric.update_state(y, logits)

            return loss_value

        epochs = 10
        for epoch in range(epochs):
            logger.info("\nStart of epoch %d" % (epoch,))
            start_time = time.time()

            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = train_step(self, x_batch_train, y_batch_train)

                # Log every 200 batches.
                if step % 20 == 0:
                    logger.info(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(loss_value))
                    )
                    logger.info("Seen so far: %d samples" % ((step + 1) * batch_size))

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            logger.info("Training acc over epoch: %.4f" % (float(train_acc),))

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            loss_list = []
            acc_list = []
            for x_batch_val, y_batch_val in val_dataset:
                loss, acc = self.model.evaluate(x_batch_val, y_batch_val, verbose=0)
                loss_list.append(loss)
                acc_list.append(acc)

            val_acc = sum(acc_list) / len(acc_list)
            val_loss = sum(loss_list) / len(loss_list)

            logger.info("Validation acc: %.4f" % (float(val_acc),))
            logger.info("Validation loss: %.4f" % (float(val_loss),))
            logger.info("Time taken: %.2fs" % (time.time() - start_time))

        return self.model