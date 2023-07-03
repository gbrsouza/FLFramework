from src.data.dataset import Dataset
from src.utils.dataset_tools import split_to_fl_simulator

import os
import cv2
from tensorflow.keras import datasets

def gen_cifar(num_clients, dest_path) -> None:
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    splited_dataset_list = split_to_fl_simulator(x_train, y_train, num_clients)

    for i in range(num_clients):
        x, y = splited_dataset_list[i]
        for j in range(len(x)):
            filename = os.path.join(dest_path, "client"+str(i), class_names[y[j][0]], "img"+str(j)+".jpg")
            cv2.imwrite(filename, x[j])

if __name__ == "__main__":
    dest_path = os.path.normpath("datasource/cifar")
    gen_cifar(5, dest_path)