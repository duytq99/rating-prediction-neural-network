import tensorflow as tf
import keras
from keras import layers
from sklearn.neighbors import NearestNeighbors

def RMSE( y_true, y_pred):
    """
    Compute root mean square error
    :param y_true: the true output
    :param y_pred: the predicted output
    :return: root mean square error
    """
    in_range1 = tf.less(y_pred, 0.5)
    y_pred = tf.where(in_range1, tf.math.multiply(tf.ones_like(y_pred),0.5), y_pred) 
    in_range2 = tf.greater(y_pred, 5)
    y_pred = tf.where(in_range2, tf.math.multiply(tf.ones_like(y_pred),5), y_pred)

    e = tf.math.subtract(y_true, y_pred)
    se = tf.square(e)  
    mse = tf.reduce_sum(se) / tf.math.count_nonzero(y_true, dtype='float64') 
    rmse = tf.math.sqrt(mse)
    return rmse  # root mean square error


def define_model():
    user_genome = keras.Input(shape=(1128,))
    X = layers.Dense(512, activation='relu')(user_genome)
    X = layers.Dense(256, activation='relu')(X)

    movie_genome = keras.Input(shape=(1128,))
    Y = layers.Dense(512, activation='relu')(movie_genome)
    Y = layers.Dense(256, activation='relu')(Y)

    concatenated = layers.Concatenate(axis=1)([X, Y])
    Z = layers.Dense(256, activation='relu')(concatenated)
    Z = layers.Dense(128, activation='relu')(Z)
    outputs = layers.Dense(1, activation='relu')(Z)

    model = keras.Model(inputs=[user_genome, movie_genome], outputs=outputs, name='recomendation')
    return model

def load_model(path, custom_objects={"RMSE": RMSE}):
    model = keras.models.load_model(path, custom_objects={"RMSE": RMSE})
    return model