# ========== #
#  Libraries #
# ========== #
from __future__ import print_function, division

import datetime
import os
import time
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

from dataset_handler import load_dataset,load_and_scale_image, load_and_scale_depth, load_and_scale_depth_test, generate_depth_maps_eigen_split
from model import build_generator, build_discriminator
from evaluation import compute_errors

# ================= #
#  Global Variables #
# ================= #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default

tf.logging.set_verbosity(tf.logging.ERROR)  # TODO: comment this line

plt.ion()

def args_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="Choose program mode (train or test).", default="train")
    parser.add_argument("-e", "--epoch", type=int, help="Define train epochs number.", default=50)
    parser.add_argument("-b", "--batch", type=int, help="Define batch size.", default=16)
    parser.add_argument("-p", "--plot", type=int, help="Define plot interval (epochs number).", default=1)
    parser.add_argument("-t", "--test", type=str, help="Choose kitti depth or eigen to test (kitti or eigen).", default="kitti")
    parser.add_argument("-d", "--max", type=float, help="Set depth max value.", default=85.0)
    parser.add_argument("-a","--mask",type=str,help="Set depth mask (50, 80 or None).",default="None")
    return parser.parse_args()

# ====== #
#  Train #
# ====== #
def train(args):
    # Defines Input shape
    img_rows, img_cols, channels, channels_depth = 256, 256, 3, 1

    img_shape = (img_rows, img_cols, channels)
    depth_shape = (img_rows, img_cols, channels_depth)

    # Sets Training Variables
    epochs = args.epoch
    batch_size = args.batch
    sample_interval = args.plot
    max_depth = args.max

    # --------
    # Dataset
    # --------
    train_images, train_labels, _, _, _, _ = load_dataset()

    # ----------
    # Optimizer
    # ----------
    optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

    # Build and compile discriminator
    discriminator = build_discriminator(img_shape, depth_shape)
    discriminator.summary()
    discriminator.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    # Build generator
    generator = build_generator(img_shape, channels_depth)
    generator.summary()

    # Input images B and their depth maps A
    img_A = tf.keras.Input(shape=depth_shape)
    img_B = tf.keras.Input(shape=img_shape)

    # Generate fake version of A from B
    fake_A = generator(img_B)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = discriminator([fake_A, img_B])

    combined = tf.keras.Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
    combined.summary()
    combined.compile(loss=['mse', 'mae'],
                     loss_weights=[1, 100],
                     optimizer=optimizer)

    start_time = datetime.datetime.now()

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / 2 ** 4)
    patch2 = int(img_cols / 2 ** 4)
    disc_patch = (patch, patch2, 1)

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    def sample_images():
        # os.makedirs('images/%s' % dataset_name, exist_ok=True)
        r, c = 1, 2

        # imgs_A, imgs_B = data_loader.load_data(batch_size=3, is_testing=True)
        imgs_B = load_and_scale_image(train_images[0])
        imgs_A = load_and_scale_depth(train_labels[0])
        fake_A = generator.predict(imgs_B)

        gen_imgs = np.concatenate([fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Input', 'Prediction', 'Ground Truth']
        fig, axs = plt.subplots(r, c)

        cax0 = axs[0].imshow(np.squeeze(gen_imgs[0], axis=2) * max_depth)
        fig.colorbar(cax0, ax=axs[0])
        axs[0].set_title(titles[1])

        cax1 = axs[1].imshow(np.squeeze(gen_imgs[1], axis=2) * max_depth)
        fig.colorbar(cax1, ax=axs[1])
        axs[1].set_title(titles[2])

        plt.show()
        plt.pause(20.0)
        plt.close('all')

    numSamples = len(train_images)

    print('Training...')

    for epoch in range(epochs):
        batch_start = 0
        batch_end = batch_size
        for batch in tqdm(range((len(train_images) // batch_size) + 1)):

            limit = batch_end

            if limit > numSamples:
                limit = numSamples
                batch_start = numSamples - batch_size

            imgs_B = np.concatenate(list(map(load_and_scale_image, train_images[batch_start:limit])), 0)
            imgs_A = np.concatenate(list(map(load_and_scale_depth, train_labels[batch_start:limit])), 0)

            # print(imgs_A.shape)
            # print(imgs_B.shape)

            batch_start += batch_size
            batch_end += batch_size

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time

            # sample_images()

        # Plot the progress
        print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                d_loss[0], 100 * d_loss[1],
                                                                                g_loss[0],
                                                                                elapsed_time))

        data_log = list(zip(epoch, epochs, d_loss[0], 100 * d_loss[1], g_loss[0], elapsed_time))
        np.savetxt('/log/data_log_%s.txt' % epoch,data_log,delimiter='\t',fmt='%s')

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            sample_images()
            generator.save_weights('/log/weights_generator_%s.h5' % epoch)
            discriminator.save_weights('/log/weights_discriminator_%s.h5' % epoch)
            combined.save_weights('/log/weights_combined_%s.h5' % epoch)

    print("Training completed.")

# ===== #
#  Test #
# ===== #
def test(args):
    def imageLoader(image_filenames):

        numSamples = len(image_filenames)

        while True:

            batch_start = 0
            batch_end = 1

            while batch_start < numSamples:

                limit = min(batch_end,numSamples)

                X_batch = np.concatenate(list(map(load_and_scale_image,image_filenames[batch_start:limit])),0)

                yield (X_batch)

                if ((limit + 1) <= numSamples):
                    batch_start += 1
                    batch_end += 1

                else:
                    del X_batch
                    batch_start = 0
                    batch_end = 1

    _, _, test_images_kitti_depth, test_labels_kitti_depth, test_images_eigen, test_labels_eigen = load_dataset()

    # --------
    # Model
    # --------
    model = build_generator()
    model.summary()

    print('Testing...')

    # -----------------------
    # Load generator weights
    # -----------------------
    # model.load_weights('weights_generator_bce.h5')

    if args.test == "kitti":
        y_pred = model.predict_generator(imageLoader(test_images_kitti_depth),steps=len(test_images_kitti_depth))
    elif args.test == "eigen":
        y_pred = model.predict_generator(imageLoader(test_images_eigen),steps=len(test_images_eigen))
    else:
        raise SystemError

    # Rescale depth maps 0 - 1
    y_pred = 0.5 * y_pred + 0.5

    y_pred = y_pred * args.max

    y_pred = y_pred[:,0,:,:,0]

    y_pred = np.expand_dims(y_pred,-1)

    print(y_pred.shape)
    print(y_pred.max())

    for i in range(5):
        plt.subplot(1,5,i + 1)
        plt.imshow(y_pred[i,:,:,0])
        plt.colorbar()
    plt.show()

    y_pred_final = []
    for i in range(len(y_pred)):
        y_pred_final.append(cv2.resize(y_pred[i,:,:,0],(1242,375),interpolation=cv2.INTER_LINEAR))

    y_pred_final = np.expand_dims(y_pred_final,-1)

    print(y_pred_final.shape)
    print(np.array(y_pred_final).max())

    for i in range(5):
        plt.subplot(1,5,i + 1)
        plt.imshow(y_pred_final[i,:,:,0])
        plt.colorbar()
    plt.show()

    imask_50 = np.where(y_pred_final < 50.0,np.ones_like(y_pred_final),np.zeros_like(y_pred_final))
    imask_80 = np.where(y_pred_final < 80.0,np.ones_like(y_pred_final),np.zeros_like(y_pred_final))
    pred_50 = np.multiply(y_pred_final,imask_50)
    pred_80 = np.multiply(y_pred_final,imask_80)

    for i in range(5):
        plt.subplot(1,5,i + 1)
        plt.imshow(pred_50[i,:,:,0])
        plt.colorbar()
    plt.show()

    for i in range(5):
        plt.subplot(1,5,i + 1)
        plt.imshow(pred_80[i,:,:,0])
        plt.colorbar()
    plt.show()

    print(np.array(pred_50).max())
    print(np.array(pred_80).max())

    depth_batch = []
    if args.test == "kitti":
        for i in test_labels_kitti_depth:
            depth = load_and_scale_depth_test(i)
            depth_batch.append(depth)
    elif args.test == "eigen":
        gt_depths = generate_depth_maps_eigen_split()
        depth_batch = []
        for i in gt_depths:
            depth = cv2.resize(i,(1242,375),interpolation=cv2.INTER_LINEAR)
            depth_batch.append(depth)
    else:
        raise SystemError

    depth_batch = np.expand_dims(depth_batch,-1)
    depth_batch = depth_batch.astype(np.float32)

    print(depth_batch.shape)
    print(depth_batch.max())

    for i in range(5):
        plt.subplot(1,5,i + 1)
        plt.imshow(depth_batch[i,:,:,0])
        plt.colorbar()
    plt.show()

    num_test_images = len(y_pred_final)

    rms = np.zeros(num_test_images,np.float32)
    log_rms = np.zeros(num_test_images,np.float32)
    abs_rel = np.zeros(num_test_images,np.float32)
    sq_rel = np.zeros(num_test_images,np.float32)
    a1 = np.zeros(num_test_images,np.float32)
    a2 = np.zeros(num_test_images,np.float32)
    a3 = np.zeros(num_test_images,np.float32)

    print('Computing metrics...')
    for i in tqdm(range(num_test_images)):
        try:
            if args.mask == "50":
                pred_depth = pred_50[i,:,:,0]
            elif args.mask == "80":
                pred_depth = pred_80[i,:,:,0]
            elif args.mask == "None":
                pred_depth = y_pred_final[i,:,:,0]
            else:
                raise SystemError

            gt_depth = depth_batch[i,:,:,0]

            pred_depth[pred_depth < (10.0 ** -3.0)] = 10.0 ** -3.0
            pred_depth[pred_depth > (80.0)] = 80.0

            mask = gt_depth > 0

            abs_rel[i],sq_rel[i],rms[i],log_rms[i],a1[i],a2[i],a3[i] = compute_errors(gt_depth[mask],pred_depth[mask])

        except IndexError:
            break

    print("abs_rel:")
    print(abs_rel.mean())
    print("sq_rel:")
    print(sq_rel.mean())
    print("rms_rel:")
    print(rms.mean())
    print("log_rms:")
    print(log_rms.mean())
    print("a1:")
    print(a1.mean())
    print("a2:")
    print(a2.mean())
    print("a3:")
    print(a3.mean())

    print("Testing completed.")


# ===== #
#  Main #
# ===== #
if __name__ == '__main__':
    args = args_handler()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise SystemError
