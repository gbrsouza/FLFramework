import os
import csv
import numpy as np
import tensorflow as tf

def encode_single_sample(self, img_path, img_height=32, img_width=32):
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

def read_csv_file(self, path):
    content_readed = {}
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            content_readed[row['ID']] = row['Class']
    return content_readed

def read_dataset (self, code):
    """read all images and labels from a dataset to tensorflow format

    Args:
        code (str): The dataset code define in the DATASETS variable
    """
    # read reference file
    reference_file = DATASETS[code]['ref']
    print("reading the reference file", reference_file)
    reference_file = self.read_csv_file(reference_file)

    # read dataset files path
    dataset_path = DATASETS[code]['data']
    print("reading the dataset file", dataset_path)
    dataset_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    print("dataset size", len(dataset_files))

    # mapping labels
    labels = []
    dataset = []
    for elem in dataset_files:
        dataset.append(os.path.join(dataset_path, elem))
        labels.append(reference_file[elem])
    print("mapping labels done!")

    # read images paths to tensorflow format
    class_names_map = DATASET_CLASSES_NAMES[code]
    print("classes names", class_names_map.values)
    images = np.array([self.encode_single_sample(x) for x in dataset])
    labels = np.array([class_names_map[x] for x in labels])

    return images, labels