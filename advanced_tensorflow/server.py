from pathlib import Path

import flwr as fl
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

from typing import List, Optional, Tuple, Union, Dict
import numpy as np

from models import load_model

def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = load_model("cnn", (64, 64, 3), 2)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.3,
        # fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        certificates=(
            Path(".cache/certificates/ca.crt").read_bytes(),
            Path(".cache/certificates/server.pem").read_bytes(),
            Path(".cache/certificates/server.key").read_bytes(),
        ),
    )

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/firestation_detector',
        seed=123,
        shuffle=True,
        batch_size=None,
        image_size=(64, 64)
    )

    server_dataset = train_ds.take(1000) 
    x_val, y_val = tuple(zip(*server_dataset))
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        #"local_epochs": 1 if server_round < 2 else 1,
        "local_epochs": 5
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()