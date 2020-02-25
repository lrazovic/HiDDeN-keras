# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.backend import sqrt, sum, square
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from PIL import Image
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,\
    Conv2D, Flatten
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D,\
    Dense, Reshape, GlobalAveragePooling2D
import string
import random

BATCH_SIZE = 128
KERNEL_SIZE = 3
LATENT_DIM = 16


def generate_random_messages(N):
    messages = list()
    for _ in range(N):
        s = []
        text = "Hello, World!"
        binary_message = string_to_binary(text)
        for c in binary_message:
            s.append(float(c))
        s = np.asarray(s)
        messages.append(s)
    return np.asarray(messages)


def model_encoder(inputs, encoder_filters, input_messages, N):
    x = inputs
    _, H, W, C = inputs.shape
    print(N, H, W, C)

    # 4 ConvBnReLU
    for filters in encoder_filters:
        x = Conv2D(filters=filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    # Phase 2
    """
       Here I'm concateneting msg, original image and conv_image
       from the previous layer.
       At the end of the for x_batch will contain all the images concatened.
    """

    for i in range(N):
        expanded_message = tf.expand_dims(input_messages[i], axis=0)
        b = tf.constant([H, W, 1], tf.int32)
        expanded_message = tf.convert_to_tensor(
            expanded_message, dtype=tf.float32)
        expanded_message = tf.tile(expanded_message, b)
        x2 = tf.concat([expanded_message, x[i], inputs[i]], -1)
        if i == 0:
            x_batch = x2
        else:
            x_batch = tf.concat([x_batch, x2], 0)

    print(x2.shape)
    img = x_batch[0]
    plt.imshow(img)
    plt.show()

    # Phase 3
    # ConvBNReLU 5
    encoded_images = Conv2D(64,
                            kernel_size=KERNEL_SIZE,
                            strides=1,
                            padding='same')(x)
    encoded_images = BatchNormalization(axis=1)(encoded_images)
    encoded_images = Activation("relu")(encoded_images)

    # Final Convolutonial Layer, no padding
    encoded_images = Conv2D(C, 1, padding='same',
                            strides=1)(encoded_images)

    # Shape info needed to build Decoder Model
    shape = K.int_shape(encoded_images)

    # Generate the latent vector (WHY ?)
    # x = Flatten()(x)
    # latent = Dense(LATENT_DIM, name='latent_vector')(x)
    # Instantiate Encoder Model
    encoder = Model([inputs, input_messages], encoded_images, name='encoder')
    return (encoder, shape)


def model_decoder(encoded_images, decoder_filters, L):
    # 7 ConvBnReLU
    x = encoded_images
    for filters in decoder_filters:
        x = Conv2D(filters=filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    # Last ConvBNReLU with L filters
    x = Conv2D(filters=L,
               kernel_size=KERNEL_SIZE,
               padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Activation('sigmoid', name='decoder_output')(x)
    decoder = Model(encoded_images, outputs, name='decoder')
    return decoder


def euclidean_distance_loss(y_true, y_pred):
    return sqrt(sum(square(y_pred - y_true), axis=-1))


def string_to_binary(string):
    # UTF-8 Encoding
    return ''.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    # MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    plt.imshow(x_train[0])
    plt.show()
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    N = len(x_train)
    # Network parameters
    input_shape = (image_size, image_size, 1)  # H*W*C
    # Encoder/Decoder number of CNN layers and filters per layer
    encoder_filters = [64, 64, 64, 64]
    decoder_filters = [64, 64, 64, 64, 64, 64, 64]
    messages = generate_random_messages(N)
    # Build the Autoencoder Model

    # First build the Encoder Model
    L = messages.shape[1]
    message_shape = (1, L)
    inputs = Input(shape=input_shape, name='encoder_input')
    input_messages = Input(shape=message_shape)
    # Instantiate Encoder Model
    (encoder, shape) = model_encoder(inputs, encoder_filters, input_messages, L)
    # encoder.summary()

    # Build the Decoder Model
    # latent_inputs = Input(shape=(LATENT_DIM,), name='decoder_input')
    shape = (image_size, image_size, 1)
    encoded_images = Input(shape, name='decoder_input')
    # Instantiate Decoder Model
    decoder = model_decoder(encoded_images, decoder_filters, L)

    # Autoencoder = Encoder + Decoder
    # Instantiate Autoencoder Model
    # autoencoder = Model(inputs=encoder, outputs=decoder, name='autoencoder')

    # autoencoder.summary()

    decoder.compile(loss=euclidean_distance_loss, optimizer='adam')

    # Train the autoencoder
    decoder.fit([x_train, messages],
                messages,
                epochs=1,
                batch_size=BATCH_SIZE)

x_decoded = decoder.predict([x_train, messages])

print(x_decoded[0])
