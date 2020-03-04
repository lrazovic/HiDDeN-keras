from tensorflow.keras.backend import sqrt, sum, square


def image_distortion_loss(y_true, y_pred):
    return sqrt(sum(square(y_pred - y_true), axis=1))/28*28


def message_distortion_loss(y_true, y_pred):
    return sqrt(sum(square(y_pred - y_true), axis=1))/96
