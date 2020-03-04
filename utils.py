import numpy as np
import string
import random


def generate_random_text(alphabet, length):
    return ''.join(random.choice(alphabet) for i in range(length))


def generate_random_messages(N):
    messages_binary = []
    message_string = []
    letters = string.ascii_lowercase
    for _ in range(N):
        s = []
        text = generate_random_text(letters, 12)
        message_string.append(text)
        binary_message = string_to_binary(text)
        for bit in binary_message:
            s.append(float(bit))
        s = np.asarray(s)
        messages_binary.append(s)
    return (message_string, np.asarray(messages_binary))


def string_to_binary(string):
    # UTF-8 Encoding
    binary_message = ''.join('{:08b}'.format(b) for b in string.encode('utf8'))
    return binary_message
