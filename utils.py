import string
import random
from const import *
import numpy as np
import tensorflow as tf
# Generate a random String with a fixed length using the passed alphabet


def generate_random_text(alphabet, length):
    return ''.join(random.choice(alphabet) for i in range(int(length)))

# Convert a string to an UTF8 binary representation


def string_to_binary(string):
    binary_message = ' '.join('{:08b}'.format(b)
                              for b in string.encode('utf8'))
    return binary_message

# Generate a random array of binary messages


def generate_random_messages(N):
    messages_binary = []
    message_string = []
    letters = string.ascii_lowercase
    for _ in range(N):
        s = []
        text = generate_random_text(letters, MESSAGE_LENGTH/8)
        message_string.append(text)
        binary_message = string_to_binary(text)
        binary_message = binary_message.replace(" ", "")
        for bit in binary_message:
            s.append(float(bit))
        s = np.asarray(s, dtype='float32')
        messages_binary.append(s)
    return (message_string, np.asarray(messages_binary, dtype='float32'))

# Round every element to 0 or 1


def round_predicted_message(predicted_message):
    rounded_message = ''
    for num in predicted_message:
        if(float(num) > 0.5):
            rounded_message += '1'
        else:
            rounded_message += '0'
    return rounded_message

# Count bit error between the original binary message
# and the predicted one


def count_errors(original_message, predicted_message):
    original_message = round_predicted_message(original_message)
    count = 0
    for i in range(len(original_message)):
        if original_message[i] != predicted_message[i]:
            count += 1
    return count

# Log in base 10 using Tensorflow


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
