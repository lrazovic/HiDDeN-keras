from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D
from const import *
import tensorflow as tf


def model_encoder(inputs, encoder_filters, input_messages):
    x = inputs
    _, H, W, C = inputs.shape

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
    expanded_message = tf.expand_dims(input_messages, axis=1)
    expanded_message = tf.expand_dims(expanded_message, axis=1)
    b = tf.constant([1, H, W, 1], tf.int32)
    expanded_message = tf.convert_to_tensor(
        expanded_message, dtype=tf.float32)
    expanded_message = tf.tile(expanded_message, b)
    x2 = tf.concat([expanded_message, x, inputs], axis=-1)
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
                            strides=1)(encoded_images)

    # encoder = Model([inputs, input_messages], encoded_images, name='encoder')
    return encoded_images
