import tensorflow as tf
from utils import OO_softmax

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def OO_masked_softmax_cross_entropy(B, Y, mask, alpha, beta, f_u):

    Z = OO_softmax(B, alpha, beta, f_u)
    loss = -tf.reduce_sum(Y * tf.math.log(Z), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)
