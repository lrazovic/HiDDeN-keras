import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# import matplotlib.pyplot as plt


def conv_bn_relu(input_channel=64):
    model = tf.keras.Sequential(name="conv_bn_relu")
    model.add(layers.ZeroPadding2D(padding=(1, 1)))
    # 50 images, 16x16 pixels and 1 channels
    # https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
    model.add(layers.Conv2D(input_channel,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    return model


def encoder(channel):
    model = tf.keras.Sequential()
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    # other code
    input_channel = (64 + 30 + channel)
    model.add(conv_bn_relu(input_channel=input_channel))
    model.add(layers.Conv2D(channel, (1, 1), strides=1,
                            padding="valid"))  # "valid" = 0 in keras
    return model


def string_to_binary(string):
    return ' '.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    st = "Hello, World"
    binary = string_to_binary(st)
    print(st)
    print(binary)
    encoder = encoder(channel=1)
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = np.expand_dims(train_images, axis=1)
    test_images = np.expand_dims(test_images, axis=1)
    encoder.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    encoder.fit(train_images, train_labels, epochs=10)
    '''
    encoder.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    encoder.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc =
    encoder.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)
    '''
