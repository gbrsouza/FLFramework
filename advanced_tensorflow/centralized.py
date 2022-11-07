import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

from models import load_model

def read_dataset():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/hospital_detector',
        shuffle=True,
        batch_size=None,
        image_size=(64, 64)
    )

    test_dataset = train_ds.take(1000) 
    train_dataset = train_ds.skip(1000)

    x_train, y_train = tuple(zip(*train_dataset))
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_test, y_test = tuple(zip(*test_dataset))
    x_test, y_test = np.array(x_test), np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def evaluate(model, x_test, y_test):

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)


def main() -> None:
    model = load_model("vgg16", (64, 64, 3), 2)
    (x_train, y_train), (x_test, y_test) = read_dataset()
    history = model.fit(
        x_train,
        y_train,
        32,
        10,
        validation_split=0.1,
    )
    evaluate(model, x_test, y_test)


if __name__ == "__main__":
    main()