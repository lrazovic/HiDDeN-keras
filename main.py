from const import *
from utils import *
from aae import HIDDEN
import matplotlib.pyplot as plt
from data_loader import load_data

if __name__ == "__main__":
    # Take some infos from the dataset
    (train_generator, test_generator, input_shape) = load_data()
    (N, H, W, C) = (train_generator.n,
                    input_shape[0], input_shape[1], input_shape[2])
    # Generate random messages as input of the encoder
    (_, messages) = generate_random_messages(N)
    L = messages.shape[1]
    epochs = 100
    print(f'{N} images, {H} x {W} x {C}')
    print(f"Message length: {L}")
    # Create the network
    network = HIDDEN(H, W, C, L, "adam")
    # Train the network
    network.train(epochs, train_generator, messages)
    (plain_test_messages, test_messages) = generate_random_messages(SIZE_TEST)
    network.predict(test_generator, test_messages, plain_test_messages, 1)
    errors = []
    i = 0
    for msg in network.decoded_msg:
        rpm = round_predicted_message(msg)
        tpm = round_predicted_message(test_messages[i])
        err = count_errors(tpm, rpm)
        errors.append(err)
        i += 1
    print(f'{sum(errors)/1000}/{network.message_length}')
    plt.axis('off')
    plt.imshow(np.squeeze(network.decoded_img[3]))
    plt.show()
    plt.imshow(np.squeeze(test_generator[0][0][3]))
    plt.show()
