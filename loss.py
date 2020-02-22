from tensorflow.keras.backend import sqrt, sum, square


def euclidean_distance_loss(y_true, y_pred):
    return sqrt(sum(square(y_pred - y_true), axis=-1))
