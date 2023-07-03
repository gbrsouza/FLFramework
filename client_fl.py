import tensorflow as tf
import flwr as fl
import argparse, sys
from model import GenericModel
from flframework_client import FlFrameworkClient

parser=argparse.ArgumentParser()

parser.add_argument("--CLIENT_NUMBER")
parser.add_argument("--EPOCHS")
parser.add_argument("--BATCH_SIZE")

args=parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

def main() -> None:

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/cifar/client{args.CLIENT_NUMBER}',
        seed=123,
        validation_split=0.1,
        batch_size=int(args.BATCH_SIZE),
        image_size=(32, 32),
        subset="training"
    )
    # generic_model = GenericModel().getModel()
    # generic_model.build(input_shape=(256, 256, 3))
    generic_model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    generic_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    client = FlFrameworkClient(generic_model, train_ds, int(args.EPOCHS))

    fl.client.start_numpy_client(
        server_address="0.0.0.0:8080",
        client=client
    )


if __name__ == "__main__":

    main()