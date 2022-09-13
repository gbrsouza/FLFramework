from src.reader.face_reader import FaceReader
from src.utils.dataset_tools import split_to_fl_simulator
from src.models.cnn import CNN
from src.models.vgg16 import LocalVGG16
from src.models.mobilenet import MobileNet
from src.actors.server import Server
from src.actors.client import Client
from src.federated_learning.fedavg import FedAvg
from src.federated_learning.fedprox import FedProx
from src.data.dataset import Dataset
from src.utils.evaluator import evaluate_model

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import os
import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

if __name__ == "__main__":
    #MobileNet('saved_models/MobileNet/global-model/checkpoint')
    # preparing dataset
    face_source = os.path.abspath('datasource/age-detection/Train')
    face_labels = os.path.abspath('datasource/age-detection/train.csv')
    reader = FaceReader(face_source, face_labels)

    data, labels = reader.read_dataset()
    dataset = Dataset((data, labels))
    dataset.equalize_data()
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.90, validation=True)
    logger.info('labels size: %s; data size: %s ', len(labels), len(data))
 
    # network_size = 2
    # logger.info('simulating FL network with %s clients', network_size)
    # splited_dataset_list = split_to_fl_simulator(x_train, y_train, network_size)

    # starting models
    #model_path = os.path.abspath('saved_models/vgg16/global-model/checkpoint')
    global_model = MobileNet('saved_models/MobileNet/global-model/checkpoint')
    global_model.create_model()
    global_model.train_model(Dataset((x_valid, y_valid)))

    # # # starting clients
    # clients = [
    #     Client(CNN('saved_models/cnn/client1/checkpoint'), Dataset(splited_dataset_list[0]), FedProx(global_model)), 
    #     Client(CNN('saved_models/cnn/client2/checkpoint'), Dataset(splited_dataset_list[1]), FedProx(global_model))
    # ]

    # #using federated leaning
 
    # models = []
    # for step, client in enumerate(clients):
    #     logger.info('training client %s', step)
    #     client.run(30)
    #     models.append(client.get_local_model())
    
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

    
