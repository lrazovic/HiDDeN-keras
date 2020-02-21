from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, ZeroPadding2D
import tensorflow as tf
import numpy as np


def model_encoder(input_imgs, input_message, N):

    [_, H, W, C] = input_imgs.shape
    print("N: ", N)
    print("C:", C)
    print("H:", H)
    print("W:", W)

    # Phase 1
    # Default data_format is "channels_last"

    # ConvBNReLU 1
    x = ZeroPadding2D(padding=(1))(input_imgs)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='valid', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 2
    x = ZeroPadding2D(padding=(1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 3
    x = ZeroPadding2D(padding=(1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 4
    x = ZeroPadding2D(padding=(1))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', strides=1)(x)
    x = BatchNormalization(axis=1)(x)

    # Phase 2
    """
       Here I'm concateneting msg, original image and conv_image
       from the previous layer.
       At the end of the for x_batch will contain all the images concatened.
    """

    for i in range(N):

        #expand message
        expanded_message = np.expand_dims(input_message, axis=0)
        expanded_message = np.expand_dims(expanded_message, axis=0)
        b = tf.constant([H, W,1], tf.int32)
        expanded_message = tf.convert_to_tensor(expanded_message, dtype=tf.float32)
        expanded_message = tf.tile(expanded_message, b)



        x2 = tf.concat([expanded_message, x[i], input_imgs[i]], -1)


        if i == 0:
            x_batch = x2

        else:
            x_batch = tf.concat([x_batch, x2], 0)

    # Phase 3
    # ConvBNReLU 5
    encoded_images = ZeroPadding2D(padding=(1))(x)
    encoded_images = Conv2D(64, (3, 3), activation='relu',
                            padding='valid', strides=1)(encoded_images)
    encoded_images = BatchNormalization(axis=1)(encoded_images)

    # Final Convolutonial Layer, no padding
    encoded_images = Conv2D(C, (1, 1), padding='same',
                            strides=1)(encoded_images)

    return encoded_images
