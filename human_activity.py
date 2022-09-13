from src.reader.human_activity_reader import HActivityReader
from src.data.dataset import Dataset
from src.models.cnn import CNN
from sklearn.preprocessing import LabelEncoder

import os
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


if __name__ == "__main__":
    # 1 - Read dataset
    source = os.path.abspath('datasource/HAR')
    reader = HActivityReader(source)
    data, labels = reader.read_dataset()

    # encoder labels to integer
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    # here we have an array of iamges paths 
    # 2 - Transform and split dataset
    dataset = Dataset((data, labels))
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(train_size=0.90, validation=True)
    logger.info('labels size: %s; data size: %s ', len(labels), len(data))

    # create a pre-trained global model
    global_model = CNN("")
    global_model.create_slim_model()
    global_model.train_model_in_batch(data=x_valid, labels=y_valid, epochs=10)