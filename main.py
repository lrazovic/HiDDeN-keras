import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from encoder import model_encoder
from decoder import model_decoder

BATCH_SIZE = 64
IMG_SIZE = 128  # All images will be resized to 192x192


def load_dataset():
    datadir = 'dataset'
    input_shape = ()
    train_datagen = ImageDataGenerator(rescale=1./255)

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
    # print("Image input %s" % str(input_shape))
    print('Loaded %d training samples.' % (num_samples))
    return train_generator


def model(channel):
    model_encoder()  # Missing parameters
    model_decoder()  # Missing parameters


def string_to_binary(string):
    return ' '.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    st = "Hello, World"
    binary = string_to_binary(st)
    train_generator = load_dataset()
    model = model(channel=3)  # Missing parameters
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"],
    )
    history = model.fit(train_generator, epochs=5, verbose=1)
    model.summary()
