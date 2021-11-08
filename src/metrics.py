import tensorflow as tf

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
    mse = tf.reduce_sum(se) / tf.math.count_nonzero(y_true, dtype='float32') 
    rmse = tf.math.sqrt(mse)
    return rmse  # root mean square error