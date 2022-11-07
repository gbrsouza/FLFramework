import flwr as fl
import tensorflow as tf
from model import GenericModel

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def main() -> None:

    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/cifar/client4',
        validation_split=0.9,
        seed=123,
        batch_size=32,
        image_size=(32, 32),
        subset="training"
    )
    model.fit(train_ds, epochs=15)

    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(256, 256, 3), weights=None, classes=2
    # )
    # # model.build(input_shape=(256, 256, 3))
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )


def get_evaluate_fn(model):

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/cifar/client4',
        validation_split=0.9,
        seed=123,
        batch_size=16,
        image_size=(32, 32),
        subset="training"
    )

    def evaluate(server_round, parameters ,config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(train_ds)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()