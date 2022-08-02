from src.reader.face_reader import FaceReader

import logging as log
log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

if __name__ == "__main__":
    face_source = 'datasource/age-detection/Train'
    face_labels = 'datasource/age-detection/train.csv'
    reader = FaceReader(face_source, face_labels)

    data, labels = reader.read_dataset()

    logger('labels size: %s; data size: %s ', len(labels), len(data))