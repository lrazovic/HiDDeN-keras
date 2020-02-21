from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization,\
    GlobalAveragePooling2D, Dense


def adversary(encoded_imgs, cover_imgs):
    # ConvBNReLU 1
    x = ZeroPadding2D(padding=(1))(encoded_imgs)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(encoded_imgs)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 2
    x = ZeroPadding2D(padding=(1))(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    # ConvBNReLU 3
    x = ZeroPadding2D(padding=(1))(x)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)

    # The activation volume is averaged over spatial dimensions.
    x = GlobalAveragePooling2D()(x)

    # Dense Layer with two output units.
    encoded_or_not = Dense(2)(x)
    return encoded_or_not
