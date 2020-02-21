import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from encoder import model_encoder
from decoder import model_decoder
import numpy as np
#import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG_SIZE = 128  # All images will be resized to 128x128


def load_dataset():
    datadir = 'dataset'
    input_shape = ()
    train_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        directory=datadir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode=None,
    )

    num_samples = train_generator.n
    print('Loaded %d training samples.' % (num_samples))
    return train_generator


def string_to_binary(string):
    return ' '.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    st = "Hello, World"
    binary_message = string_to_binary(st)
    message_length = len(binary_message)
    print("The original message is '{}'".format(st))
    print("The binary message is '{}'".format(binary_message))
    print("The length of the binary message is '{}'".format(message_length))

    # train_generator = load_dataset()
    # input_img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # N = train_generator.n
    binary_message = np.array(binary_message)

    # Using MNIST as Dataset
    (X_train, _), (X_test, _) = mnist.load_data()
    shape_x = 28
    shape_y = 28
    input_img = layers.Input(shape=(shape_x, shape_y, 1))
    N = len(X_train)

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = X_train.reshape(-1, shape_x, shape_y, 1)
    X_test = X_test.reshape(-1, shape_x, shape_y, 1)

    encoded_images = model_encoder(
        input_img, binary_message, N)
    decoded_messages = model_decoder(encoded_images, message_length)

    autoencoder = Model(input_img, decoded_messages)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.summary()
    autoencoder.fit(X_train, X_train,
                    epochs=1,
                    shuffle=True,)
