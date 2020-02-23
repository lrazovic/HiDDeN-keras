from tensorflow.keras.layers import Activation, Dense, Input, \
    BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from const import *


def model_encoder(inputs, encoder_filters):
    x = inputs
    for filters in encoder_filters:
        x = Conv2D(filters=filters,
                   kernel_size=KERNEL_SIZE,
                   strides=1,
                   padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation("relu")(x)
    x = Conv2D(1, 1, padding='same', strides=1)(x)
    # Shape info needed to build Decoder Model
    shape = K.int_shape(x)

    # Generate the latent vector
    x = Flatten()(x)
    latent = Dense(LATENT_DIM, name='latent_vector')(x)

    # Instantiate Encoder Model
    encoder = Model(inputs, latent, name='encoder')
    return (encoder, shape)
