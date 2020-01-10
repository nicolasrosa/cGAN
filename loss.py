import tensorflow as tf

def tf_mask_out_invalid_pixels(tf_pred, tf_labels):
    # Identify Pixels to be masked out.
    tf_idx = tf.where(tf_labels > 0.0)  # Tensor 'idx' of Valid Pixel values (batchID, idx)

    # Mask Out Pixels without depth values
    tf_valid_pred = tf.gather_nd(tf_pred, tf_idx)
    tf_valid_labels = tf.gather_nd(tf_labels, tf_idx)

    return tf_valid_pred, tf_valid_labels

# -------------------- #
#  Mean Squared Error  #
# -------------------- #
def tf_mse_loss(y_true, y_pred):

    # Mask Out
    y_pred, y_true = tf_mask_out_invalid_pixels(y_pred,y_true)

    tf_npixels = tf.cast(tf.size(y_pred), tf.float32)

    # Loss
    mse = tf.div(tf.reduce_sum(tf.square(y_pred - y_true)), tf_npixels)

    return mse

# -------------------- #
#  Mean Absolute Error  #
# -------------------- #
def tf_mae_loss(y_true, y_pred):

    # Mask Out
    y_pred,y_true = tf_mask_out_invalid_pixels(y_pred,y_true)

    tf_npixels = tf.cast(tf.size(y_pred), tf.float32)

    # Loss
    mae = tf.div(tf.reduce_sum(tf.abs(y_pred - y_true)), tf_npixels)

    return mae