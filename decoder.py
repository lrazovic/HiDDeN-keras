from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D,\
    Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from const import *


def model_decoder(latent_inputs, decoder_filters, shape, L):
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

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

    #x = GlobalAveragePooling2D()(x)
    outputs = Activation('sigmoid', name='decoder_output')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder
