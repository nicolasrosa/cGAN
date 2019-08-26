import tensorflow as tf


class cGAN:
    def __init__(self, img_shape, depth_shape):
        self.img_shape = img_shape
        self.depth_shape = depth_shape

    @staticmethod
    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = tf.keras.layers.LeakyReLU(alpha=0.2)(d)
        if bn:
            d = tf.keras.layers.BatchNormalization(momentum=0.8)(d)
        return d

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
        u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = tf.keras.layers.Dropout(dropout_rate)(u)
        u = tf.keras.layers.BatchNormalization(momentum=0.8)(u)
        u = tf.keras.layers.Concatenate()([u, skip_input])
        return u

    # ================= #
    #  U-Net Generator  #
    # ================= #
    def build_generator(self, gf=64):
        """Args: Number of filters in the first layer of G"""

        # Image input
        d0 = tf.keras.Input(shape=self.img_shape)

        # Downsampling
        d1 = self.conv2d(d0, gf, bn=False)
        d2 = self.conv2d(d1, gf * 2)
        d3 = self.conv2d(d2, gf * 4)
        d4 = self.conv2d(d3, gf * 8)
        d5 = self.conv2d(d4, gf * 8)
        d6 = self.conv2d(d5, gf * 8)
        d7 = self.conv2d(d6, gf * 8)

        # Upsampling
        u1 = self.deconv2d(d7, d6, gf * 8)
        u2 = self.deconv2d(u1, d5, gf * 8)
        u3 = self.deconv2d(u2, d4, gf * 8)
        u4 = self.deconv2d(u3, d3, gf * 4)
        u5 = self.deconv2d(u4, d2, gf * 2)
        u6 = self.deconv2d(u5, d1, gf)

        u7 = tf.keras.layers.UpSampling2D(size=2)(u6)

        output_img = tf.keras.layers.Conv2D(self.depth_shape[2], kernel_size=4, strides=1, padding='same',
                                            activation='linear')(u7)  # TODO: 'tanh' ou 'linear'?

        return tf.keras.Model(d0, output_img)

    # =================== #
    #  CNN Discriminator  #
    # =================== #
    def build_discriminator(self, df=64):
        """Args: Number of filters in the first layer of D"""

        img_A = tf.keras.Input(shape=self.depth_shape)
        img_B = tf.keras.Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = tf.keras.layers.Concatenate(axis=-1)([img_A, img_B])

        d1 = self.conv2d(combined_imgs, df, bn=False)
        d2 = self.conv2d(d1, df * 2)
        d3 = self.conv2d(d2, df * 4)
        d4 = self.conv2d(d3, df * 8)
        # d5 = self.conv2d(d4,df * 16)

        validity = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return tf.keras.Model([img_A, img_B], validity)
