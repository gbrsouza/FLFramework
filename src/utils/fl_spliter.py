from os import listdir
from os.path import isfile, join
import random


def split_to_fl_simulator (dataset_path, size):
    """This function split a dataset in other with same size
    to simulate a federated learning network

    Args:
        dataset_path (string): the dataset path
        size (int): the number of datasets to split
    """

    # read all files from dataset
    allfiles = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
    splited_dataset_size = int(len(allfiles)/size)
    print("total dataset size:", len(allfiles))
    print("size of each splited dataset", splited_dataset_size)

    # shuffle the file list
    random.shuffle(allfiles)

    # split datasets
    chunks = [allfiles[x:x+splited_dataset_size] for x in range(0, len(allfiles), splited_dataset_size)]
    return chunks[:size]