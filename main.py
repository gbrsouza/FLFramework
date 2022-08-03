from src.reader.face_reader import FaceReader
from src.utils.dataset_tools import split_to_fl_simulator
from src.models.cnn import CNN
from src.actors.server import Server
from src.actors.client import Client
from src.federated_learning.fedavg import FedAvg

import os
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

if __name__ == "__main__":
    # preparing dataset
    face_source = os.path.abspath('datasource/age-detection/Train')
    face_labels = os.path.abspath('datasource/age-detection/train.csv')
    reader = FaceReader(face_source, face_labels)

    data, labels = reader.read_dataset()
    logger('labels size: %s; data size: %s ', len(labels), len(data))
    
    network_size = 2
    logger.info('simulating FL network with %s clients', network_size)
    splited_dataset_list = split_to_fl_simulator(data, labels, network_size)

    # starting models
    model_path = os.path.abspath('saved_models/cnn/checkpoint')
    global_model = CNN(model_path)
    global_model.create_model()

    # starting clients
    clients = [
        Client(global_model ,splited_dataset_list[0], FedAvg()), 
        Client(global_model ,splited_dataset_list[1], FedAvg())
    ]

    #using federated leaning
    cnt = 0
    models = []
    for client in clients:
        logger.info('training client %s', cnt)
        client.run(5)
        models.append(client.get_local_model())
        cnt += 1

    server = Server()
    new_model = server.aggregate_models(models, global_model)

    
