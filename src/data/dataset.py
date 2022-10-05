import numpy as np
import logging as log

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class Dataset:
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __init__(self, tuple):
        self.data = tuple[0]
        self.labels = tuple[1]

    def split_data(self, train_size=0.9, validation=False, shuffle=True):
        size = len(self.data)
        indices = np.arange(size)

        if shuffle:
            np.random.shuffle(indices)

        if validation:
            test_validation_size = (1.0-train_size)/2.0
            
            train_samples = int(size * train_size)
            test_sample = train_samples + int(size * test_validation_size)
            # print(type(indices))
            # print(indices[:10])

            x_train, y_train = self.data[indices[:train_samples]], self.labels[indices[:train_samples]]
            x_test, y_test = self.data[indices[train_samples:test_sample]], self.labels[indices[train_samples:test_sample]]
            x_valid, y_valid = self.data[indices[test_sample:]], self.labels[indices[test_sample:]]
            return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
        else: 
            train_samples = int(size * train_size)
            x_train, y_train = self.data[indices[:train_samples]], self.labels[indices[:train_samples]]
            x_test, y_test = self.data[indices[train_samples:]], self.labels[indices[train_samples:]]
            return (x_train, y_train), (x_test, y_test), (None, None)

    def equalize_data(self):
        unique, counts = np.unique(self.labels, return_counts=True)
        result = np.column_stack((unique, counts)) 
        logger.info('classes count before balancing dataset')
        print(result)

        class_size = min(counts)
        # separe classes
        mapped_indices = {}
        for label in unique:
            mapped_indices[label] = []

        for ind, label in enumerate(self.labels):
            mapped_indices[label[0]].append(ind)

        X_under, y_under = [], []
        for label in unique:
            indices = mapped_indices[label]
            np.random.shuffle(indices)

            x, y = self.data[indices[:class_size]], self.labels[indices[:class_size]]
            X_under.extend(x)
            y_under.extend(y)

        unique, counts = np.unique(y_under, return_counts=True)
        result = np.column_stack((unique, counts)) 
        logger.info('classes count after balancing dataset')
        print(result)
        print(len(X_under))

        self.data = np.array(X_under)
        self.labels = np.array(y_under)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    