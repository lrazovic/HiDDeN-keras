from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    AveragePooling2D


def model_decoder(input_imgs, input_message):

    [N, C, H, W] = input_imgs.size()
    message_len = len(input_message)

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(message_length, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    output_message = AveragePooling2D(pool_size=(message_len, message_len),
                                      strides=1,
                                      padding='same',
                                      data_format="channels_first")(x)
    return output_message
