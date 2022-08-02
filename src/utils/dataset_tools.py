import tensorflow as tf
import numpy as np

def encode_single_sample(img_path, img_height=32, img_width=32):
    try:
        img = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        return img
    
    except Exception as e:
        print('file_path', img_path)
        print(e)
        return e

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