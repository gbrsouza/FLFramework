import os
import logging as log

import socket
import pickle
from tensorflow.keras.models import model_from_json

from src.reader.face_reader import FaceReader
from src.utils.dataset_tools import split_to_fl_simulator
from src.models.cnn import CNN
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



log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == "__main__":


    soc = socket.socket()
    soc.connect(("localhost", 5454))

    received_data = b''
    while str(received_data)[-2] != '.':
        data = soc.recv(1237928)
        received_data += data

    received_data = pickle.loads(received_data)
    model = model_from_json(received_data['model_json'])
    logger.info('Global model received from central server')
    global_model = CNN('saved_models/cnn/global-model/checkpoint')
    global_model.create_model()
    global_model.model = model
    global_model.model.load_weights("model.h5")

    face_source = os.path.abspath('datasource/age-detection/Train')
    face_labels = os.path.abspath('datasource/age-detection/train.csv')
    reader = FaceReader(face_source, face_labels)

    data, labels = reader.read_dataset()
    dataset = Dataset((data, labels))
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.90, validation=True)
    logger.info('labels size: %s; data size: %s ', len(labels), len(data))

    splited_dataset_list = split_to_fl_simulator(x_train, y_train, 2)

    client = Client(CNN('saved_models/cnn/client1/checkpoint'), Dataset(splited_dataset_list[0]), FedProx(global_model)) 

    logger.info('Training client')
    client.run(1)

    logger.info('Metrics for client')
    acc ,pre, rec, matrix = evaluate_model(x_test, y_test, client.get_local_model())
    logger.info('acc: %.4f, pre: %.4f, rec: %.4f', acc, pre, rec)

    data = {
        'model_name': 'client_1',
        'weigths': client.get_local_model().get_weights()
    }
    response = pickle.dumps(data)
    soc.sendall(response)

    logger.info('Client model sent to server!')


    soc.close()

