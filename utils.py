import numpy as np


def string_to_binary(string):
    # UTF-8 Encoding
    return ' '.join(format(ord(x), 'b') for x in string)


def generate_random_messages(N):
    messages = list()
    for _ in range(N):
        s = []
        text = "Hello, World!"
        binary_message = string_to_binary(text)
        for c in binary_message:
            s.append(float(c))
        s = np.asarray(s)
        messages.append(s)
    return np.asarray(messages)
