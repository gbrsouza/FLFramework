from src.reader.face_reader import FaceReader
from src.utils.dataset_tools import split_to_fl_simulator
from src.models.cnn import CNN

import os
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

if __name__ == "__main__":
    face_source = os.path.abspath('datasource/age-detection/Train')
    face_labels = os.path.abspath('datasource/age-detection/train.csv')
    reader = FaceReader(face_source, face_labels)

    data, labels = reader.read_dataset()
    logger('labels size: %s; data size: %s ', len(labels), len(data))
    
    network_size = 2
    logger.info('simulating FL network with %s clients', network_size)
    splited_dataset_list = split_to_fl_simulator(data, labels, network_size)

    model_path = os.path.abspath('saved_models/cnn/checkpoint')
    global_model = CNN(model_path)
    global_model.create_model()

    
