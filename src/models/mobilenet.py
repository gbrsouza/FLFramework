from unittest import TextTestResult
from src.models.abstract_model import Model
from src.data.dataset import Dataset
from src.utils.dataset_tools import load_data
from tensorflow.keras import layers, models
from sklearn import metrics
from src.utils.evaluator import evaluate_model


import tensorflow as tf
import logging as log
import numpy as np

log.basicConfig(level=log.NOTSET)
logger = log.getLogger("logger")

class MobileNet(Model):
    def __init__(self, model_path, type='multiclass'):
        super().__init__(model_path, type)
        self.create_model()

    # overriding abstract method
    def create_model(self, to_save=False):
        preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
        model = tf.keras.applications.MobileNetV3Large(input_shape = (128,128,3,),include_top=False,weights='imagenet')

        # Set model.trainable to false, this prevents changing weights of pre-trained model.
        model.trainable = False

        # Adding custom layers to the pre-trained model.
        inputs = tf.keras.Input(shape=(128, 128, 3))
        x = preprocess_input(inputs)
        x = model(x, training=True)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1000)(x)
        x = tf.keras.layers.Dense(600)(x)
        x = tf.keras.layers.Dense(600)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(400)(x)
        x = tf.keras.layers.Dense(400)(x)
        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(50)(x)
        x = tf.keras.layers.Dense(50)(x)
        x = tf.keras.layers.Dense(20)(x)
        x = tf.keras.layers.Dense(20)(x)
        outputs = tf.keras.layers.Dense(3)(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

        self.model = model

    def train_model(self, dataset: Dataset):
        (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = dataset.split_data(validation=True)

        trainDS = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        testDS = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        validationDS = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

        AUTOTUNE = tf.data.AUTOTUNE
        trainDS = (trainDS
            .map(load_data, num_parallel_calls=AUTOTUNE)
            .batch(64)
            .prefetch(AUTOTUNE)
        )

        trainDS = trainDS.map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
        ).cache(
        ).map(
            lambda image, label: (tf.image.per_image_standardization(image), label)
        ).map(
            lambda image, label: (tf.image.random_contrast(image, lower=0.4, upper=0.6), label)
        ).map(
            lambda image, label: (tf.image.random_brightness(image, max_delta = 0.4), label)
        ).map(
            lambda image, label: (tf.image.random_hue(image, max_delta = 0.4), label)
        ).map(
            lambda image, label: (tf.image.random_saturation(image, lower=0.4, upper=0.6), label)
        ).shuffle(
            1000
        ).repeat(2)

        AUTOTUNE = tf.data.AUTOTUNE
        validationDS = (validationDS
            .map(load_data, num_parallel_calls=AUTOTUNE)
            .batch(64)
            .prefetch(AUTOTUNE)
        )

        validationDS = validationDS.map(
            lambda image, label: (tf.image.random_flip_left_right(image), label)
        ).cache(
        ).map(
            lambda image, label: (tf.image.per_image_standardization(image), label)
        ).map(
            lambda image, label: (tf.image.random_contrast(image, lower=0.4, upper=0.6), label)
        ).map(
            lambda image, label: (tf.image.random_brightness(image, max_delta = 0.4), label)
        ).map(
            lambda image, label: (tf.image.random_hue(image, max_delta = 0.4), label)
        ).map(
            lambda image, label: (tf.image.random_saturation(image, lower=0.4, upper=0.6), label)
        ).shuffle(
            1000
        ).repeat(2)

        testDS = (testDS
            .map(load_data, num_parallel_calls=AUTOTUNE)
            .batch(64)
            .prefetch(AUTOTUNE)
        )

        testDS = testDS.map(
            lambda image, label: (tf.image.per_image_standardization(image), label)
        )

        # Training the model
        self.model.fit(trainDS, epochs=17, validation_data=validationDS)

        # self.model.fit(x_train, y_train, epochs=20, 
        #             validation_data=(x_test, y_test))
        
        
        test_images = [load_data(path) for path in x_test]
        evaluate_model(test_images, y_test, self.model)

        # Predict probabilites of test data.
        # probabilities = self.model.predict(testDS)
        # # print(probabilities)
        # # Create classes from predictions
        # predictions = np.argmax(probabilities,axis=1)
        # actual_values = y_test
        # print(actual_values)
        # print(predictions)
        

        # print("balanced accuracy:   %0.3f" % metrics.balanced_accuracy_score(actual_values, predictions))
        # print("accuracy:   %0.3f" % metrics.accuracy_score(actual_values, predictions))





        # self.save_model()