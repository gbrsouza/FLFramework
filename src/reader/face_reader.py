from src.reader.abstract_reader import Reader
from src.utils.dataset_tools import encode_single_sample

import numpy as np
import logging as log
import csv
import os

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

DATASET_CLASSES_NAMES = {
    'YOUNG': np.array([0], dtype=np.uint8), 
    'MIDDLE': np.array([1], dtype=np.uint8), 
    'OLD' : np.array([2], dtype=np.uint8)
}

class FaceReader(Reader):

    def __init__(self, source, labels):
        super().__init__(source)
        self.labels = labels

    def read_csv_file(self, path):
        """read a csv from face dataset

        Args:
            path (string): the dataset path

        Returns:
            dict: a dict cotained the file id and the label
        """
        content_readed = {}
        with open(path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                content_readed[row['ID']] = row['Class']
        return content_readed

    # overriding abstract method
    def read_dataset(self):
        logger.info("reading labels of face dataset from path %s", self.labels)
        labels_map = self.read_csv_file(self.labels)

        logger.info("reading the dataset file from path %s", self.source)
        data = [f for f in os.listdir(self.source) if os.path.isfile(os.path.join(self.source, f))]
        logger.info("dataset size %s", len(data))

        # mapping labels
        labels, dataset = [], []
        for elem in data:
            dataset.append(os.path.join(self.source, elem))
            labels.append(labels_map[elem])
        logger.info("labels mapped!")

        # read images paths to tensorflow format
        logger.info("classes names %s", DATASET_CLASSES_NAMES.values)
        images = np.array(dataset)
        labels = np.array([DATASET_CLASSES_NAMES[x] for x in labels])

        return images, labels



