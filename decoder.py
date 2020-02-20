from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    GlobalAveragePooling2D, Dense


def model_decoder(encoded_imgs, input_message):

    [N, C, H, W] = encoded_imgs.size()
    message_len = len(input_message)

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(message_length, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = GlobalAveragePooling2D(data_format="channels_first")(x)
    output_message = Dense(message_len*message_len, activation="sigmoid")(x)
    return output_message
