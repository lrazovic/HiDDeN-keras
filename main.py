import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# import matplotlib.pyplot as plt

BATCH_SIZE = 64
IMG_SIZE = 128  # All images will be resized to 192x192


def load_dataset():
    datadir = './dataset'
    # testset = datadir+'/test/'
    input_shape = ()
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    train_generator = train_datagen.flow_from_directory(
        directory=datadir,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode=None,
    )

    num_samples = train_generator.n
    input_shape = train_generator.image_shape
    print("Image input %s" % str(input_shape))
    print('Loaded %d training samples.' % (num_samples))
    return train_generator


def conv_bn_relu():
    model = tf.keras.Sequential(name="conv_bn_relu")
    model.add(layers.ZeroPadding2D(padding=(1, 1)))
    # https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
    model.add(layers.Conv2D(64,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same"))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.Activation('relu'))
    return model


def model_one(channel):
    model = tf.keras.Sequential()
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    model.add(conv_bn_relu())
    return model


def model_two(channel):
    model = tf.keras.Sequential()
    # other code
    channel = (64 + 30 + channel)
    model.add(conv_bn_relu())
    # "valid" = 0 in keras
    model.add(layers.Conv2D(channel, (1, 1), strides=1, padding="valid"))
    return model


def encoder(channel):
    return model_one(channel=channel)
    # model_two()


def string_to_binary(string):
    return ' '.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    st = "Hello, World"
    binary = string_to_binary(st)
    train_generator = load_dataset()
    encoder = encoder(channel=1)
    encoder.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"],
    )
    history = encoder.fit_generator(train_generator, epochs=10, verbose=1)
    encoder.summary()
    '''
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
