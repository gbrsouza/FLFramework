import argparse
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

import flwr as fl
import csv
from datetime import datetime
import numpy as np

from models import load_model
import time


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CustomClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, id):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.id = id
        self.round = 0
        self.create_result_file()

    def create_result_file(self):
        row = "round & loss & accuracy & precision & time \\\\ \hline \n"
        now = datetime.now()
        postfix = now.strftime('%Y%m%d%H%M')

        file_path = "./results/client0" + str(self.id) + "-" + postfix

        self.result_file = file_path

        with open(self.result_file, 'w') as f:
            f.write(row)
             

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        start = time.time()
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        end = time.time()
        self.elapse = end - start

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, pre = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)

        num_examples_test = len(self.x_test)

        row = str(self.round) + " & " + str(round(loss,3)) + " & " + str(round(accuracy,2)) + " & " + str(round(pre,2)) + " & " + str(round(self.elapse, 0)) + " \\\\ \\hline \n"
        with open(self.result_file, 'a') as f:
            f.write(row)

        self.round = self.round + 1

        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 5), required=True)
    parser.add_argument("--epochs", type=int, choices=range(1, 30), required=True)
    parser.add_argument("--clients", type=int, choices=range(1, 5), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = load_model("efinet", (64, 64, 3), 1)

    # Load a subset of dataset to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition, args.clients)

    # Start Flower client
    client = CustomClient(model, x_train, y_train, x_test, y_test, args.partition)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int, clients: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(clients)

    print("CLIENT %s --- Lendo dados para treino", idx)
    # read dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/firestation_detector',
        shuffle=True,
        batch_size=None,
        image_size=(64, 64)
    )

    print("CLIENT %s --- Fazendo treino", idx)

    test_dataset = train_ds.take(2000) 
    train_dataset = train_ds.skip(2000)

    x_train, y_train = tuple(zip(*train_dataset))
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_test, y_test = tuple(zip(*test_dataset))
    x_test, y_test = np.array(x_test), np.array(y_test)

    s_client_data_train = int(len(x_train)/clients)
    s_client_data_test = int(len(x_test)/clients)
    #x_test, y_test = tf.keras.datasets.cifar10.load_data()
    
    return (
        x_train[idx * s_client_data_train : (idx + 1) * s_client_data_train],
        y_train[idx * s_client_data_train : (idx + 1) * s_client_data_train],
    ), (
        x_test[idx * s_client_data_test : (idx + 1) * s_client_data_test],
        y_test[idx * s_client_data_test : (idx + 1) * s_client_data_test],
    )


if __name__ == "__main__":
    main()
