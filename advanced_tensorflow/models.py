import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.applications.inception_v3 import InceptionV3


def load_model(id: str, input: (int, int, int), output: int):
    if id == "cnn":
        model = models.Sequential()
        model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))

        model.add(layers.Dense(output, activation="sigmoid"))
        model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])

    elif id == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input)
        base_model.trainable = False

        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(64, activation='relu')
        dense_layer_2 = layers.Dense(32, activation='relu')
        dense_layer_3 = layers.Dense(16, activation='relu')

        prediction_layer = layers.Dense(output, activation="sigmoid")

        model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            dense_layer_3,
            prediction_layer
        ])

        model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])

    elif id == "inception":

        base_model  = InceptionV3(weights="imagenet", include_top=False, input_shape=input)
        base_model.trainable = False

        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(1024, activation='relu')
        # dense_layer_2 = layers.Dense(64, activation='relu')
        # dense_layer_3 = layers.Dense(32, activation='relu')
        dropout_layer_2 = layers.Dropout(0.2)

        prediction_layer = layers.Dense(output, activation="sigmoid")

        model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer_1,
            # dense_layer_2,
            # dense_layer_3,
            # dropout_layer_2,
            prediction_layer
        ])

        model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])

    elif id == 'efinet':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input, weights="imagenet")
        base_model.trainable = False

        # pooling_layer = layers.GlobalAveragePooling2D()
        # norm_layer = layers.BatchNormalization()
        # dropout_layer = layers.Dropout(0.4)

        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(64, activation='relu')
        dense_layer_2 = layers.Dense(32, activation='relu')
        dense_layer_3 = layers.Dense(16, activation='relu')

        prediction_layer = layers.Dense(output, activation="sigmoid")

        # model = models.Sequential([
        #     base_model,
        #     pooling_layer,
        #     norm_layer,
        #     dropout_layer,
        #     prediction_layer
        # ])

        model = models.Sequential([
            base_model,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            dense_layer_3,
            prediction_layer
        ])

        model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])

    else:
        model = tf.keras.applications.EfficientNetB0(
            input_shape=input, weights=None, classes=output
        )
        model.compile("adam", "binary_crossentropy", metrics=["accuracy", tf.keras.metrics.Precision()])
    
    return model