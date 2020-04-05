from const import *
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, BatchNormalization,\
    Conv2D, Input, GaussianNoise, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from loss import *
from utils import *
import numpy as np


class HIDDEN():
    # This is the class of the entire network
    def __init__(self, height, width, channel, message_length, optimizer):
        self.message_length = message_length  # L on the paper
        self.H = height  # H on the paper
        self.W = width  # W on the paper
        self.C = channel  # C on the paper
        self.image_shape = (self.H, self.W, self.C)
        print("Build models...")
        self._build_encoder_model()
        self._build_noise_layer_model("identity")
        self._build_decoder_model()
        self._build_discriminator_model()
        self._build_and_compile_network(optimizer)

    def _build_encoder_model(self):
        # Build the encoder
        print("Building Encoder...")
        input_images = Input(shape=self.image_shape, name='encoder_input')
        input_messages = Input(shape=self.message_length,
                               name='input_messages')
        # Phase 1
        x = input_images
        # Applying 4 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64, 64]:
            x = Conv2D(filters=filters,
                       kernel_size=KERNEL_SIZE,
                       strides=1,
                       padding='same',
                       use_bias=False)(x)
            x = BatchNormalization(-1)(x)
            x = Activation("relu")(x)

        # Phase 2
        expanded_message = tf.expand_dims(input_messages, axis=1)
        expanded_message = tf.expand_dims(expanded_message, axis=1)
        a = tf.constant([1, self.H, self.W, 1], tf.int32)
        expanded_message = tf.convert_to_tensor(
            expanded_message, dtype=tf.float32)
        # Replicating the message H*W times
        expanded_message = tf.tile(expanded_message, a)
        # Concatenate messages and images channel-wise
        x2 = tf.concat([expanded_message, x, input_images], axis=-1)

        # Phase 3
        # Latest Conv-BN-ReLU block with 64 output filters
        encoded_images = Conv2D(64,
                                kernel_size=KERNEL_SIZE,
                                strides=1,
                                padding='same',
                                use_bias=False)(x2)
        encoded_images = BatchNormalization(-1)(encoded_images)
        encoded_images = Activation("relu")(encoded_images)

        # Final Convolutonial Layer with 1 x 1 kernel and C output filters
        encoded_images = Conv2D(self.C, 1, padding='same',
                                strides=1)(encoded_images)

        self.encoder_model = Model(
            [input_images, input_messages], encoded_images, name='encoder')

    def _build_noise_layer_model(self, name):
        # Function that applies the noise layer to the image
        print("Building Noise Layer...")
        input_images = Input(shape=self.image_shape, name='noise_input')
        if name == "identity":
            self.noise_layer_model = Model(
                input_images, input_images, name='noise')
        elif name == "gaussian":
            x = GaussianNoise(2)(input_images)
            self.noise_layer_model = Model(input_images, x, name='noise')
        elif name == "dropout":
            x = Dropout(0.3)(input_images)
            self.noise_layer_model = Model(input_images, x, name='noise')

    def _build_decoder_model(self):
        # Build the decoder
        print("Building Decoder Generator...")
        input_images = Input(shape=self.image_shape, name='decoder_input')
        x = input_images
        # Applying 7 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64, 64, 64, 64, 64]:
            x = Conv2D(filters,
                       kernel_size=KERNEL_SIZE,
                       strides=1,
                       padding='same',
                       use_bias=False)(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)

        # Last ConvBNReLU with L filters
        x = Conv2D(self.message_length,
                   kernel_size=KERNEL_SIZE,
                   padding='same',
                   use_bias=False)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)

        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer with L units
        x = Dense(self.message_length)(x)
        self.decoder_model = Model(input_images, x, name='decoder')

    def _build_discriminator_model(self):
        # build the adversary
        input_images = Input(shape=self.image_shape, name='adversary_input')
        x = input_images
        # Applying 3 Conv-BN-ReLU blocks with 64 output filters
        for filters in [64, 64, 64]:
            x = Conv2D(filters,
                       kernel_size=KERNEL_SIZE,
                       strides=1,
                       padding='same')(x)
            x = BatchNormalization(axis=-1)(x)
            x = Activation("relu")(x)
        # Average Pooling over all spatial dimensions
        x = GlobalAveragePooling2D()(x)
        # Final linear layer to classify the image
        adversary_output = Dense(2, activation="softmax")(x)
        self.discriminator_model = Model(
            input_images, adversary_output, name='discriminator')

    def _build_and_compile_network(self, optimizer):
        self.discriminator_model.compile(
            loss=discriminator_loss, optimizer="adam")
        # We will only train the Encoder and the Decoder
        self.discriminator_model.trainable = False
        print("Conecting models...")

        images = Input(shape=self.image_shape, name='input')
        messages = Input(shape=self.message_length, name='messages')
        encoder_output = self.encoder_model([images, messages])
        noise_output = self.noise_layer_model(encoder_output)
        decoder_output = self.decoder_model(noise_output)
        discriminator_output = self.discriminator_model(encoder_output)
        # The final network: Encoder + Noise + Decoder + Adversary
        self.network = Model([images, messages], [
                             noise_output, decoder_output, discriminator_output], name='hidden')

        # Compile all the network
        self.network.compile(loss=["mse", message_distortion_loss, adversary_loss],
                             # The relative weights of the losses, lambda_i and lambda_g
                             loss_weights=[0.7, 1, 0.001],
                             optimizer=optimizer)

    # Train on batch the entire network
    def train(self, epochs, train_images, train_messages):
        for epoch in range(epochs + 1):
            batch, _ = next(train_images)
            batch_size = len(batch)
            index = np.random.randint(0, len(train_images), batch_size)
            real = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            batch_messages = train_messages[index]
            cover_images = batch
            encoded_images = self.encoder_model.predict(
                [batch, batch_messages])
            # Train the adversary
            loss_real = self.discriminator_model.train_on_batch(
                cover_images, real)
            loss_fake = self.discriminator_model.train_on_batch(
                encoded_images, fake)
            #  Train all the network
            autoencoder_loss = self.network.train_on_batch(
                [batch, batch_messages], [batch, batch_messages, real])
            print(
                f"Epoch {epoch} Autoencoder loss: {autoencoder_loss[0]}\
                    Image loss: {autoencoder_loss[1]}\
                        Message loss: {autoencoder_loss[2]},\
                            Adversary loss: {autoencoder_loss[3]}")

    # Predict on batch
    def predict(self, prediction_images, prediction_messages, plain_msg, index):
        print("Starting Prediction")
        decoded_img = []
        decoded_msg = []
        x = prediction_images
        for i in range(len(x)):
            batch, _ = next(x)
            batch_size = len(batch)
            pred_messages = prediction_messages[i *
                                                batch_size:i*batch_size + batch_size]
            (imgs, msgs, _) = self.network.predict_on_batch(
                [batch, pred_messages])
            for img in imgs:
                decoded_img.append(img)
            for msg in msgs:
                decoded_msg.append(msg)

        self.decoded_img = decoded_img
        self.decoded_msg = decoded_msg

        predicted_message = round_predicted_message(decoded_msg[index])
        original_message = round_predicted_message(prediction_messages[index])
        print("Original message as String: ", plain_msg[index])
        print("Original message in Binary: ", original_message)
        print("Predicted message in Binary: ", predicted_message)

        errors = count_errors(original_message, predicted_message)
        print(f'Errors {errors}/{self.message_length}')

    def save(self, path):
        self.network.save(path)
