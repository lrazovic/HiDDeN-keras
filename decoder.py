from tensorflow.keras.layers import Activation, Dense, Input, \
    BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from const import *


def model_decoder(latent_inputs, decoder_filters, shape):
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for filters in decoder_filters:
        x = Conv2D(filters=filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    x = Conv2D(filters=1,
               kernel_size=KERNEL_SIZE,
               padding='same')(x)
    # x = Conv2D(message_length, (3, 3), activation='relu',
    #                padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    outputs = Activation('sigmoid', name='decoder_output')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder
