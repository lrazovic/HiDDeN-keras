from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    GlobalAveragePooling2D, Dense


def model_decoder(encoded_imgs, message_length):
    # ConvBNReLU 1
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 2
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 3
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 4
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 5
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 6
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 7
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU, with L filters.
    x = Conv2D(message_length, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)

    x = GlobalAveragePooling2D()(x)
    output_message = Dense(message_length,
                           activation="sigmoid")(x)
    return output_message
