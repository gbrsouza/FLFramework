import flwr as fl
import tensorflow as tf
from model import GenericModel

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
#   except RuntimeError as e:
#     print(e)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def main() -> None:



    # model = GenericModel().getModel()

    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     f'./datasource/dog-cat/train/client03',
    #     validation_split=0.9,
    #     seed=123,
    #     batch_size=32,
    #     subset="training"
    # )
    # model.fit(train_ds, epochs=2)

    model = tf.keras.applications.EfficientNetB0(
        input_shape=(256, 256, 3), weights=None, classes=2
    )
    # model.build(input_shape=(256, 256, 3))
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
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
        f'./datasource/client03',
        validation_split=0.9,
        seed=123,
        batch_size=16,
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