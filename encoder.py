from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization


def model_encoder(input_imgs, input_messages):

    [N, C, H, W] = input_imgs.size()

    # Phase 1

    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', strides=1)(input_imgs)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=1)(x)
    x = BatchNormalization(axis=1)(x)

    # Phase 2

    # Here I'm concateneting msg, original image and conv_image from the previous layer
    # At the end of the for x_batch will contain all the images concatened

    for i in range(N):
        msg = input_messages[i].repeat(H, W, 1).permute(2, 0, 1)

        x2 = tf.concat([x[i], msg, input_imgs[i]], 1)

        if i == 0:
            x_batch = x2

        else:
            x_batch = tf.concat([x_batch, x2], 0)

    # Phase 3

    encoded_images = Conv2D(64, (3, 3), activation='relu',
                            padding='same', strides=1)(x_batch)
    encoded_images = Conv2D(C, (1, 1), padding='same',
                            strides=1)(encoded_images)

    return encoded_images
