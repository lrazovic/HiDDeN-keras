import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from encoder import model_encoder
from decoder import model_decoder
from utils import generate_random_messages
from loss import image_distortion_loss, message_distortion_loss
import matplotlib.pyplot as plt
from const import *


# MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train[:SIZE]
x_test = x_test[:SIZE]
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
(plain_messages, messages) = generate_random_messages(N)
# First build the Encoder Model
L = messages.shape[1]
print(f'L is {L}')
message_shape = (L,)
inputs = Input(shape=input_shape, name='encoder_input')
input_messages = Input(shape=message_shape)
# Instantiate Encoder Model
encoded_images = model_encoder(inputs, encoder_filters, input_messages)
# Build the Decoder Model
# Instantiate Decoder Model
decoder = model_decoder(encoded_images, decoder_filters, L)
autoencoder = Model([inputs, input_messages], [encoded_images, decoder])
autoencoder.compile(loss=[image_distortion_loss,
                          message_distortion_loss], optimizer='adam')

# Train the autoencoder
autoencoder.fit([x_train, messages],
                [x_train, messages],
                epochs=20,
                validation_split=.2,
                batch_size=BATCH_SIZE)
