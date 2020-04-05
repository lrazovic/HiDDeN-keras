from tensorflow.keras.preprocessing.image import ImageDataGenerator
from const import *


def load_data():
    # Get and process the COCO dataset
    datadir = 'dataset'
    trainingset = datadir + '/train/'
    testset = datadir + '/tester/'

    # Normalizing the train images to the range of [0., 1.]
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=trainingset,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="input",
        shuffle=False
    )

    # Normalizing the test images to the range of [0., 1.]
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=testset,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="input",
        shuffle=False
    )

    num_samples = train_generator.n
    input_shape = train_generator.image_shape

    print(f"Image input {input_shape}")
    print(f'Loaded {num_samples} training samples')
    print(f'Loaded {test_generator.n} test samples')
    return (train_generator, test_generator, input_shape)
