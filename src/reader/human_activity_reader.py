from src.reader.abstract_reader import Reader

import logging as log
import pandas as pd
import numpy as np
import os

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


class HActivityReader(Reader):
    
    def __init__(self, source):
        super().__init__(source)

    def read_dataset(self):

        source = pd.read_csv(os.path.join(self.source, "Training_set.csv"))

        data, labels = [], []

        for i in range(len(source)):
            img_path = os.path.join(self.source, "train", source["filename"][i])
            data.append(img_path)
            labels.append(source["label"][i])

        return np.array(data), np.array(labels)
        