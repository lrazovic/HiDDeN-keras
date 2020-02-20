from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    GlobalAveragePooling2D, Dense


def model_decoder(encoded_imgs, message_length):

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(message_length, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = GlobalAveragePooling2D()(x)
    output_message = Dense(message_length,
                           activation="sigmoid")(x)
    return output_message
