import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
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
    print('Loaded %d training samples.' % (num_samples))
    return train_generator


def string_to_binary(string):
    return ' '.join(format(ord(x), 'b') for x in string)


if __name__ == "__main__":
    st = "Hello, World"
    binary_message = string_to_binary(st)
    message_length = len(binary_message)
    train_generator = load_dataset()
    input_img = layers.Input(shape=(128, 128, 3))
    input_messages = layers.Input(shape=(message_length, 1))

    encoded_images = model_encoder(input_img, input_messages)

    decoded_messages = model_decoder(encoded_images, message_length)

    autoencoder = Model(input_img, decoded_messages)

    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit()  # dobbiamo vedere come passargli i vari parametri
