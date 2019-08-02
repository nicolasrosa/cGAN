import tensorflow as tf


# ================= #
#  U-Net Generator  #
# ================= #
def build_generator(img_shape, channels_depth, gf=64):
    """Args: Number of filters in the first layer of G"""

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        if bn:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
        u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = tf.keras.layers.Dropout(dropout_rate)(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = tf.keras.Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)
    d5 = conv2d(d4, gf * 8)
    d6 = conv2d(d5, gf * 8)
    d7 = conv2d(d6, gf * 8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf * 8)
    u2 = deconv2d(u1, d5, gf * 8)
    u3 = deconv2d(u2, d4, gf * 8)
    u4 = deconv2d(u3, d3, gf * 4)
    u5 = deconv2d(u4, d2, gf * 2)
    u6 = deconv2d(u5, d1, gf)

    u7 = tf.keras.layers.UpSampling2D(size=2)(u6)
    output_img = tf.keras.layers.Conv2D(channels_depth, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return tf.keras.Model(d0, output_img)


# =================== #
#  CNN Discriminator  #
# =================== #
def build_discriminator(img_shape, depth_shape, df=64):
    """Args: Number of filters in the first layer of D"""

    def d_layer(layer_input, filters, f_size=4, bn=True):
        """Discriminator layer"""
        d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        if bn:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        return d

    img_A = tf.keras.Input(shape=depth_shape)
    img_B = tf.keras.Input(shape=img_shape)

    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)
    # d5 = d_layer(d4,df * 16)

    validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return tf.keras.Model([img_A, img_B], validity)
