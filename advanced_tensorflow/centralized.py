import tensorflow as tf
from tensorflow.keras import layers, models

import numpy as np

from models import load_model
import time 


result_file = "./results/avg/hospital_unb/server-avg-hospital-efinet-centralized-5-unb.tex" 

def read_dataset():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/hospital_detector_seed123/train',
        shuffle=True,
        batch_size=None,
        image_size=(45, 45)
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f'./datasource/hospital_detector_seed123/test',
        shuffle=True,
        batch_size=None,
        image_size=(45, 45)
    )

    x_train, y_train = tuple(zip(*train_ds))
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_test, y_test = tuple(zip(*test_ds))
    x_test, y_test = np.array(x_test), np.array(y_test)

    return (x_train, y_train), (x_test, y_test)

def evaluate(model, x_test, y_test):

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    loss, acc, pre = model.evaluate(x_test, y_test, batch_size=128)
    row = "& " + str(5) + " & " + str(round(loss,3)) + " & " + str(round(acc,2)) + " & " + str(round(pre, 2)) + " & "
    with open(result_file, 'a') as f:
        f.write(row)


def main() -> None:
    model = load_model("efinet", (45, 45, 3), 1)
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
    for i in range(10):
        start = time.time()
        main()
        end = time.time()
        t = str(end-start)
        row = t + " \\\\ \n"
        with open(result_file, 'a') as f:
            f.write(row)