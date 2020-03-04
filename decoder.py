from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D,\
    Dense, Reshape, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from const import *


def model_decoder(encoded_images, decoder_filters, L):
    # 7 ConvBnReLU
    x = encoded_images
    for filters in decoder_filters:
        x = Conv2D(filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)

    # Last ConvBNReLU with L filters
    x = Conv2D(L,
               kernel_size=KERNEL_SIZE,
               padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = GlobalAveragePooling2D()(x)

    outputs = Activation('sigmoid', name='decoder_output')(x)
    # decoder = Model(encoded_images, outputs, name='decoder')
    return outputs
