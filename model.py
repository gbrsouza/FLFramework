import tensorflow as tf
from tensorflow.keras import layers, models, losses

class GenericModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.model = None
       
    def getModel(self):
        

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))


        # model = tf.keras.Sequential([
        #     tf.keras.layers.Rescaling(1./255),
        #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Conv2D(32, 3, activation='relu'),
        #     tf.keras.layers.MaxPooling2D(),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dense(2)
        # ])

        # model.compile(optimizer='adam',
        #               loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=['accuracy'])

        return self.model