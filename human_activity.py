from src.reader.human_activity_reader import HActivityReader
from src.data.dataset import Dataset

from src.models.cnn import CNN
from src.models.inceptionv3 import Inception

from src.actors.client import Client
from src.actors.server import Server
from src.utils.dataset_tools import split_to_fl_simulator
from sklearn.preprocessing import LabelEncoder
from src.utils.dataset_tools import processing_image_dataset

from src.federated_learning.fedprox import FedProx

from src.utils.evaluator import evaluate_model

import os
import logging as log
import numpy as np 

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


if __name__ == "__main__":
    # 1 - Read dataset
    source = os.path.abspath('datasource/HAR')
    reader = HActivityReader(source)
    data, labels = reader.read_dataset()
    classes = np.unique(labels, return_counts=False)

    # encoder labels to integer
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    # here we have an array of iamges paths 
    # 2 - Transform and split dataset
    dataset = Dataset((data, labels))
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.90, validation=False)
    logger.info('labels size: %s; data size: %s ', len(labels), len(data))

    # 3 - create a pre-trained global model
    # global_model = CNN("")
    # global_model.create_slim_model()
    global_model = Inception("")
    global_model.create_model()
    global_model.train_model_in_batch(data=x_train, labels=y_train, epochs=100)

    # 4 - Split dataset to simulate clients
    # network_size = 2
    # logger.info('simulating FL network with %s clients', network_size)
    # splited_dataset_list = split_to_fl_simulator(x_train, y_train, network_size)

    # 5 - starting clients
    # clients = [
    #     Client(CNN('saved_models/cnn/client1/checkpoint'), Dataset(splited_dataset_list[0]), FedProx(global_model)), 
    #     Client(CNN('saved_models/cnn/client2/checkpoint'), Dataset(splited_dataset_list[1]), FedProx(global_model))
    # ]

    # 6 - using federated leaning
 
    # models = []
    # for step, client in enumerate(clients):
    #     logger.info('training client %s', step)
    #     client.run(15)
    #     models.append(client.get_local_model())

    x_test, y_test = processing_image_dataset(x_test, y_test, (100,100))

    logger.info('Metrics for global model')
    acc ,pre, rec, matrix = evaluate_model(x_test, y_test, global_model, classes)
    logger.info('\n-------confusion matrix-------\n%s', matrix)
    logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)
    
    # for step, model in enumerate(models):
    #     logger.info('Metrics for client %s', step)
    #     acc ,pre, rec, matrix = evaluate_model(x_test, y_test, model)
    #     logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

    # server = Server()
    # weights = server.aggregate_models(models)
    # global_model.set_weights(weights)

    # acc ,pre, rec, matrix = evaluate_model(x_test, y_test, global_model)
    # logger.info('Metrics from aggregated model')
    # logger.info('\n-------confusion matrix-------\n%s', matrix)
    # logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)