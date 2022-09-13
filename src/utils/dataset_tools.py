import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array, load_img

import logging as log
log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")


def encode_single_sample(img_path, ch=3, resize=(128,128)):
    try:
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=ch)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, resize)
        return img
    
    except Exception as e:
        print('file_path', img_path)
        print(e)
        return e

def load_data(image_path, label, size=(150, 150)):
    image = encode_single_sample(image_path, 3, size)
    return (image, label)
    

def processing_img(path, label, size=(150, 150)):
    img = load_img(path, target_size=size)
    img = img_to_array(img)
    return (img, label)

def processing_image_dataset(data, labels, size=(150,150)):
    dataset = []
    for elem in data:
        dataset.append(processing_img(elem, size))
    return np.array(dataset), np.array(labels)

def split_to_fl_simulator (dataset, labels, size):
    """This function split a dataset in other with same size
        to simulate a federated learning network

    Args:
        dataset (array): The dataset with all files
        labels (array): the labels for each file in the dataset
        size (int): the number of datasets to split
    """

    # read all files from dataset
    splited_dataset_size = int(len(dataset)/size)
    print("total dataset size:", len(dataset))
    print("size of each splited dataset", splited_dataset_size)

    # shuffle the file list
    size = len(dataset)
    indices = np.arange(size)
    np.random.shuffle(indices)

    # split datasets
    chunks = [(dataset[indices[x:x+splited_dataset_size]], labels[indices[x:x+splited_dataset_size]]) for x in range(0, len(dataset), splited_dataset_size)]
    return chunks