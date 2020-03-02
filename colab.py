# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras.backend import sqrt, sum, square
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D, Flatten, Input, GlobalAveragePooling2D
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import string
import random

print(tf.__version__)

"""## Const"""

BATCH_SIZE = 128
KERNEL_SIZE = 3

"""## Utility"""


def generate_random_messages(N):
    messages = []
    for i in range(N):
        s = []
        if i == 20:
            text = "Asdfg, Ertyu?"
            binary_message = string_to_binary(text)
            binary_message = binary_message.replace(" ", "")
            for c in binary_message:
                for elem in c:
                    s.append(elem)
            s = np.asarray(s, dtype=float)
            messages.append(s)
        else:
            text = "Hello, World!"
            binary_message = string_to_binary(text)
            binary_message = binary_message.replace(" ", "")
            for c in binary_message:
                for elem in c:
                    s.append(elem)
            s = np.asarray(s, dtype=float)
            messages.append(s)
    return np.asarray(messages)


def euclidean_distance_loss(y_true, y_pred):
    return sqrt(sum(square(y_pred - y_true), axis=-1))


def string_to_binary(string):
    # UTF-8 Encoding
    return ' '.join('{:08b}'.format(b) for b in string.encode('utf8'))


"""## Encoder"""


def model_encoder(inputs, encoder_filters, input_messages, N):
    x = inputs
    _, H, W, C = inputs.shape

    # 4 ConvBnReLU
    for filters in encoder_filters:
        x = Conv2D(filters=filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    # Phase 2
    """
       Here I'm concateneting msg, original image and conv_image
       from the previous layer.
       At the end of the for x_batch will contain all the images concatened.
    """
    expanded_message = tf.expand_dims(input_messages, axis=1)
    expanded_message = tf.expand_dims(expanded_message, axis=1)
    b = tf.constant([1, H, W, 1], tf.int32)
    expanded_message = tf.convert_to_tensor(
        expanded_message, dtype=tf.float32)
    expanded_message = tf.tile(expanded_message, b)
    x2 = tf.concat([expanded_message, x, inputs], axis=-1)
    # Phase 3
    # ConvBNReLU 5
    print("Concat DONE")
    # Phase 3
    # ConvBNReLU 5
    encoded_images = Conv2D(64,
                            kernel_size=KERNEL_SIZE,
                            strides=1,
                            padding='same')(x2)
    encoded_images = BatchNormalization(axis=1)(encoded_images)
    encoded_images = Activation("relu")(encoded_images)

    # Final Convolutonial Layer, no padding
    encoded_images = Conv2D(C, 1, padding='same',
                            strides=1, kernel_regularizer=l2(0.01))(encoded_images)

    # encoder = Model([inputs, input_messages], encoded_images, name='encoder')
    return encoded_images


"""## Decoder"""


def model_decoder(encoded_images, decoder_filters, L):
    # 7 ConvBnReLU
    x = encoded_images
    for filters in decoder_filters:
        x = Conv2D(filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same',
                   kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    # Last ConvBNReLU with L filters
    x = Conv2D(L,
               kernel_size=KERNEL_SIZE,
               padding='same',
               kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)
    # decoder = Model(encoded_images, outputs, name='decoder')
    return outputs


"""## Autoencoder"""

# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[:5000]
x_test = x_test[:5000]
image_size = x_train.shape[1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
N = len(x_train)

# Network parameters
input_shape = (image_size, image_size, 1)  # H*W*C

# Encoder/Decoder number of CNN layers and filters per layer
encoder_filters = [64, 64, 64, 64]
decoder_filters = [64, 64, 64, 64, 64, 64, 64]
messages = generate_random_messages(N)
# First build the Encoder Model
L = messages.shape[1]
print(f'L is {L}')
message_shape = (L)
inputs = Input(shape=input_shape, name='encoder_input')
input_messages = Input(shape=message_shape)
# Instantiate Encoder Model
encoded_images = model_encoder(inputs, encoder_filters, input_messages, N)
# Build the Decoder Model
# Instantiate Decoder Model
decoder = model_decoder(encoded_images, decoder_filters, L)
autoencoder = Model([inputs, input_messages], decoder)
autoencoder.compile(optimizer='nadam', loss="mse")

# Train the autoencoder
autoencoder.fit([x_train, messages],
                messages,
                epochs=50,
                batch_size=BATCH_SIZE)

"""## Predictions"""


def generate_random_messages_test(N):
    messages = []
    for i in range(N):
        s = []
        text = "Asdfg, Ertyu?"
        binary_message = string_to_binary(text)
        binary_message = binary_message.replace(" ", "")
        for c in binary_message:
            for elem in c:
                s.append(elem)
        s = np.asarray(s, dtype=float)
        messages.append(s)
    return np.asarray(messages)


imgs = x_train[:2]
mess = generate_random_messages_test(2)
prediction = autoencoder.predict([imgs, mess])
print(prediction[0])

print("Original Message:")
msg = string_to_binary("Asdfg, Ertyu?")
binary_message = msg.replace(" ", "")
print(binary_message)
decoded = prediction[0]
clean_decode = []
for elem in decoded:
    if elem > 0.6:
        clean_decode.append(1)
    else:
        clean_decode.append(0)

c = ""
for elem in clean_decode:
    c += str(elem)
print("Decoded: {}".format(c))
