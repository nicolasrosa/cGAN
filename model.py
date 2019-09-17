import keras

class cGAN:
    def __init__(self, img_shape, depth_shape):
        self.img_shape = img_shape
        self.depth_shape = depth_shape

    @staticmethod
    def conv2d(layer_input, filters, f_size=4, bn=True):
        d = keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = keras.layers.LeakyReLU(alpha=0.2)(d)
        if bn:
            d = keras.layers.BatchNormalization(momentum=0.8)(d)
        return d

    @staticmethod
    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        # u = keras.layers.UpSampling2D(size=2)(layer_input)
        # u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        u = keras.layers.Conv2DTranspose(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(
            layer_input)
        if dropout_rate:
            u = keras.layers.Dropout(dropout_rate)(u)
        u = keras.layers.BatchNormalization(momentum=0.8)(u)
        u = keras.layers.Concatenate()([u, skip_input])
        u = keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        return u

    # ================= #
    #  U-Net Generator  #
    # ================= #
    def build_generator(self, gf=64):
        """Args: Number of filters in the first layer of G"""

        # Image input
        d0 = keras.Input(shape=self.img_shape)

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

        # u7 = keras.layers.UpSampling2D(size=2)(u6)

        # output_img = keras.layers.Conv2D(self.depth_shape[2], kernel_size=4, strides=1, padding='same',
        #                                     activation='linear')(u7)  # TODO: 'tanh' ou 'linear'?
        output_img = keras.layers.Conv2DTranspose(self.depth_shape[2], kernel_size=4, strides=2, padding='same',
                                                     activation='linear')(u6)

        return keras.Model(d0, output_img)

    def build_generator_resnet(self):
        """U-Net Generator"""

        d0 = keras.Input(shape=self.img_shape)

        d = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet')
        d.trainable = True
        d = d(d0)

        up0 = keras.layers.Conv2DTranspose(512, kernel_size=4, activation='relu', padding='same', strides=2)(d)
        conv0 = keras.layers.Conv2D(512, kernel_size=4, activation='relu', padding='same')(up0)

        up1 = keras.layers.Conv2DTranspose(256, kernel_size=4, activation='relu', padding='same', strides=2)(conv0)
        conv1 = keras.layers.Conv2D(256, kernel_size=4, activation='relu', padding='same')(up1)

        up2 = keras.layers.Conv2DTranspose(128, kernel_size=4, activation='relu', padding='same', strides=2)(conv1)
        conv2 = keras.layers.Conv2D(128, kernel_size=4, activation='relu', padding='same')(up2)

        up3 = keras.layers.Conv2DTranspose(64, kernel_size=4, activation='relu', padding='same', strides=2)(conv2)
        conv3 = keras.layers.Conv2D(64, kernel_size=4, activation='relu', padding='same')(up3)

        up4 = keras.layers.Conv2DTranspose(64, kernel_size=4, activation='relu', padding='same', strides=2)(conv3)
        conv4 = keras.layers.Conv2D(64, kernel_size=4, activation='relu', padding='same')(up4)

        output_img = keras.layers.Conv2D(self.depth_shape[2], kernel_size=4, padding='same', activation='linear')(conv4)

        return keras.Model(d0, output_img)

    # =================== #
    #  CNN Discriminator  #
    # =================== #
    def build_discriminator(self, df=64):
        """Args: Number of filters in the first layer of D"""

        img_A = keras.Input(shape=self.depth_shape)
        img_B = keras.Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = keras.layers.Concatenate(axis=-1)([img_A, img_B])

        d1 = self.conv2d(combined_imgs, df, bn=False)
        d2 = self.conv2d(d1, df * 2)
        d3 = self.conv2d(d2, df * 4)
        d4 = self.conv2d(d3, df * 8)
        # d5 = self.conv2d(d4,df * 16)

        validity = keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return keras.Model([img_A, img_B], validity)
