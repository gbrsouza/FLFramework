import tensorflow as tf
from src.models.inceptionv3 import Inception
from src.utils.evaluator import evaluate_model
from src.models.vgg16 import LocalVGG16
from src.models.cnn import CNN
from src.data.dataset import Dataset
from src.utils.dataset_tools import split_to_fl_simulator
from src.actors.client import Client
from src.actors.server import Server
from src.federated_learning.fedprox import FedProx
from src.federated_learning.fedavg import FedAvg

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import logging as log   
log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


if __name__ == "__main__":
    # read dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # here we have an array of iamges paths 
    # 2 - Transform and split dataset
    dataset = Dataset((train_images, train_labels))
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.70, validation=False)
    # logger.info('labels size: %s; data size: %s ', len(labels), len(data))

    global_model = CNN("")
    global_model.create_model(input_shape=(32,32,3), num_classes=10)
    global_model.train_model(Dataset((x_test, y_test)), epochs=30)

    logger.info('Metrics for global model')
    acc ,pre, rec, matrix = evaluate_model(test_images, test_labels, global_model, class_names)
    logger.info('\n-------confusion matrix-------\n%s', matrix)
    logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

    # 4 - Split dataset to simulate clients
    network_size = 3
    logger.info('simulating FL network with %s clients', network_size)
    splited_dataset_list = split_to_fl_simulator(x_train, y_train, network_size)

    # 5 - starting clients
    clients = [
        Client(CNN('saved_models/weigths/client0/checkpoint'), Dataset(splited_dataset_list[0]), FedAvg()), 
        Client(CNN('saved_models/weigths/client1/checkpoint'), Dataset(splited_dataset_list[1]), FedAvg()),
        Client(CNN('saved_models/weigths/client2/checkpoint'), Dataset(splited_dataset_list[2]), FedAvg())
    ]

    # 6 - using federated leaning
 
    models = []
    for step, client in enumerate(clients):
        logger.info('training client %s', step)
        client.run(70)
        models.append(client.get_local_model())

    for step, model in enumerate(models):
        logger.info('Metrics for client %s', step)
        model.save_model()
        acc ,pre, rec, matrix = evaluate_model(test_images, test_labels, model, class_names)
        logger.info('\n-------confusion matrix-------\n%s', matrix)
        logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

    server = Server()
    weights = server.aggregate_models(models)
    global_model.set_weights(weights)

    acc ,pre, rec, matrix = evaluate_model(test_images, test_labels, global_model, class_names)
    logger.info('Metrics for aggregated model')
    logger.info('\n-------confusion matrix-------\n%s', matrix)
    logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

