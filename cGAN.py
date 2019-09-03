# ========== #
#  Libraries #
# ========== #
from __future__ import print_function, division

import argparse
import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

from dataset_handler import load_dataset, load_and_scale_image, load_and_scale_depth, generate_depth_maps_eigen_split
from model import cGAN
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

datetime_var = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

showImages = False  # TODO: create args

# ========== #
#  Functions #
# ========== #
# noinspection PyTypeChecker
def args_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="Chooses program mode.", required=True)
    parser.add_argument("-d", "--dataset_name", type=str, help="Chooses dataset.", required=True)

    # TODO: Terminar de arrumar
    parser.add_argument("-e", "--epochs", type=int, help="Define train epochs number.", default=50)
    parser.add_argument("-b", "--batch_size", type=int, help="Define batch size.", default=4)
    parser.add_argument("-l", "--learn_rate", type=str, help="Define learning rate.", default=1e-4)
    parser.add_argument("--beta", type=str, help="Define Adam's beta.", default=0.5)

    parser.add_argument("--max_depth", type=float, help="Set depth max value.", default=85.0)
    parser.add_argument("--single_image", action='store_true', help="Train model on a single image.", default=False)

    parser.add_argument("-p", "--plot", type=int, help="Define plot interval (epochs number).", default=1)
    parser.add_argument("-t", "--test_split", type=str, help="Choose kitti depth or eigen to test (kitti or eigen).",
                        default="kitti_depth")
    parser.add_argument("-a", "--mask", type=str, help="Set depth mask (50, 80 or None).", default="None")
    return parser.parse_args()


# ====== #
#  Train #
# ====== #
def train():
    # Plot
    plt.ion()
    titles = ['Original', 'Prediction (Translated)', 'GT']
    fig, axs = plt.subplots(1, 2)

    cax0 = axs[0].imshow(np.zeros((256, 256)))
    cbar0 = fig.colorbar(cax0, ax=axs[0], fraction=0.045)
    axs[0].set_title(titles[1])

    cax1 = axs[1].imshow(np.zeros((256, 256)))
    cbar1 = fig.colorbar(cax1, ax=axs[1], fraction=0.045)
    axs[1].set_title(titles[2])
    fig.tight_layout(pad=0.2, w_pad=2.5, h_pad=None)  # Fix Subplots Spacing

    # Defines Input shape
    img_rows, img_cols, channels, channels_depth = 256, 256, 3, 1

    img_shape = (img_rows, img_cols, channels)
    depth_shape = (img_rows, img_cols, channels_depth)

    # Sets Training Variables
    dataset_name = args.dataset_name
    epochs = args.epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate
    beta = args.beta
    max_depth = args.max_depth
    sample_interval = args.plot

    # Creates output directory
    model_folder = "output/{}/{}/".format(dataset_name, datetime_var)
    os.makedirs(model_folder)
    print("\nDirectory '", model_folder, "' created.", sep='')

    # --------
    # Dataset
    # --------
    train_images, train_labels, _, _ = load_dataset(dataset_name)
    numSamples = len(train_images)

    if args.single_image:
        train_images = [train_images[0]]
        train_labels = [train_labels[0]]
        numSamples=0
        batch_size=1

    # -------------------------
    # Construct Computational
    #   Graph of Generator
    # -------------------------
    model = cGAN(img_shape, depth_shape)

    optimizer = tf.keras.optimizers.Adam(learn_rate, beta)

    # Build and compile the discriminator
    discriminator = model.build_discriminator()
    discriminator.summary()
    discriminator.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    # Build the generator
    generator = model.build_generator()
    generator.summary()

    # Input images and their conditioning images
    img_A = tf.keras.Input(shape=depth_shape)
    img_B = tf.keras.Input(shape=img_shape)

    # By conditioning on B generate a fake version of A
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

    start_time = time.time()

    # Calculate output shape of D (PatchGAN)
    patch = int(img_rows / 2 ** 4)
    patch2 = int(img_cols / 2 ** 4)
    disc_patch = (patch, patch2, 1)

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    def update_colorbar(cbar, img):
        vmin, vmax = np.min(img), np.max(img)
        cbar_ticks = np.linspace(vmin, vmax, num=7, endpoint=True)

        cbar.set_clim(vmin, vmax)
        cbar.set_ticks(cbar_ticks)
        cbar.draw_all()

    def sample_images():
        # os.makedirs('images/%s' % dataset_name, exist_ok=True)

        imgs_B = load_and_scale_image(train_images[0])
        imgs_A = load_and_scale_depth(train_labels[0], (256,256))
        fake_A = generator.predict(imgs_B)

        gen_imgs = np.concatenate([fake_A, imgs_A])
        gen_imgs[0] = np.exp(gen_imgs[0])-1
        gen_imgs[1] = np.exp(gen_imgs[1])-1

        cax0.set_data(gen_imgs[0, :, :, 0])
        update_colorbar(cbar0, gen_imgs[0, :, :, 0])

        cax1.set_data(gen_imgs[1, :, :, 0])
        update_colorbar(cbar1, gen_imgs[1, :, :, 0])

        # plt.show()
        plt.draw()
        plt.pause(0.0001)
        # plt.close('all')

    for epoch in range(epochs):
        batch_start = 0
        batch_end = batch_size

        for _ in tqdm(range((numSamples // batch_size) + 1)):

            limit = batch_end

            if limit > numSamples:
                limit = numSamples
                batch_start = numSamples - batch_size

            try:
                imgs_B = np.concatenate(list(map(load_and_scale_image, train_images[batch_start:limit])), 0)
                imgs_A = np.concatenate(list(map(load_and_scale_depth, train_labels[batch_start:limit])), 0)
            except ValueError:  # Single Image Workaround
                imgs_B = load_and_scale_image(train_images[0])
                imgs_A = load_and_scale_depth(train_labels[0])

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

            elapsed_time = time.time() - start_time

            # timer1 = -time.time()
            sample_images()
            # timer1 += time.time()
            # print(timer1)

        # Plot the progress
        print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch + 1, epochs,
                                                                                d_loss[0], 100 * d_loss[1],
                                                                                g_loss[0],
                                                                                elapsed_time))

        # If at save interval => save generated image samples
        if epoch % sample_interval == 0:
            generator.save_weights(model_folder + 'weights_generator_bce.h5')
            discriminator.save_weights(model_folder + 'weights_discriminator_bce.h5')
            combined.save_weights(model_folder + 'weights_combined_bce.h5')

    print("\nTraining completed.")


# ===== #
#  Test #
# ===== #
def test():
    # Defines Input shape
    img_rows, img_cols, channels, channels_depth = 256, 256, 3, 1

    img_shape = (img_rows, img_cols, channels)
    depth_shape = (img_rows, img_cols, channels_depth)

    # --------
    # Dataset
    # --------
    _, _, test_images_kitti_depth, test_labels_kitti_depth, test_images_eigen, test_labels_eigen = load_dataset(args.dataset_name)

    if args.test_split == 'kitti_depth':
        test_images = test_images_kitti_depth
        test_labels = test_labels_kitti_depth
    elif args.test_split == 'eigen':
        test_images = test_images_eigen
        test_labels = test_labels_eigen
    else:
        raise SystemError

    num_test_images = len(test_images)
    # num_test_images = 10

    # --------
    # Model
    # --------
    model = cGAN(img_shape, depth_shape)
    generator = model.build_generator()
    generator.summary()

    # -----------------------
    # Load generator weights
    # -----------------------
    # generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/kitti_morphological/2019-09-02_10-29-44/weights_generator_bce.h5')
    generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/weights_generator_linear4.h5')

    # ------------
    # Predictions
    # ------------
    # Generate Predictions
    print('Generating Predictions...')
    y_pred = []
    for k in tqdm(range(num_test_images)):
        imgs_B = load_and_scale_image(test_images[k])
        fake_A = generator.predict(imgs_B)
        y_pred.append(fake_A[0, :, :, 0])

    # Resize Predictions
    y_pred_up = []
    for pred in y_pred:
        pred_resized = cv2.resize(pred, (1242, 375), interpolation=cv2.INTER_LINEAR)
        y_pred_up.append(pred_resized)

    y_pred_up = np.array(y_pred_up)  # list -> np.array

    # Mask
    imask_50 = np.where(y_pred_up < 50.0, np.ones_like(y_pred_up), np.zeros_like(y_pred_up))
    imask_80 = np.where(y_pred_up < 80.0, np.ones_like(y_pred_up), np.zeros_like(y_pred_up))
    pred_50 = np.multiply(y_pred_up, imask_50)
    pred_80 = np.multiply(y_pred_up, imask_80)

    # print(np.array(pred_50).max())
    # print(np.array(pred_80).max())

    # -------------
    # Ground-Truth
    # -------------
    print('\nGenerating ground truth images...')

    gt_depths = []
    if args.test_split == "kitti_depth":
        for i in tqdm(range(num_test_images)):
            depth = load_and_scale_depth(test_labels[i], (1242, 375))
            gt_depths.append(depth[0, :, :, 0])

    elif args.test_split == "eigen":
        gt_depths = generate_depth_maps_eigen_split()

        for i in tqdm(range(num_test_images)):
            gt_depths[i] = cv2.resize(gt_depths[i], (1242, 375), interpolation=cv2.INTER_LINEAR)
    else:
        raise SystemError

    gt_depths = np.array(gt_depths)  # list -> np.array

    # Plot
    if showImages:
        for i in range(num_test_images):
            plt.figure(1)
            plt.imshow(y_pred[i])
            plt.figure(2)
            plt.imshow(pred_50[i])
            plt.figure(3)
            plt.imshow(pred_80[i])
            plt.figure(4)
            plt.imshow(gt_depths[i])
            plt.pause(0.0001)
            plt.draw()

    # Free Memory
    del y_pred

    # --------
    # Metrics
    # --------
    rms = np.zeros(num_test_images, np.float32)
    log_rms = np.zeros(num_test_images, np.float32)
    abs_rel = np.zeros(num_test_images, np.float32)
    sq_rel = np.zeros(num_test_images, np.float32)
    a1 = np.zeros(num_test_images, np.float32)
    a2 = np.zeros(num_test_images, np.float32)
    a3 = np.zeros(num_test_images, np.float32)

    print('Computing metrics...')
    for i in tqdm(range(num_test_images)):
        if args.mask == "50":
            pred_depth = pred_50[i]
        elif args.mask == "80":
            pred_depth = pred_80[i]
        elif args.mask == "None":
            pred_depth = y_pred_up[i]
        else:
            raise SystemError

        gt_depth = gt_depths[i]

        pred_depth[pred_depth < (10.0 ** -3.0)] = 10.0 ** -3.0
        pred_depth[pred_depth > 80.0] = 80.0

        mask = gt_depth > 0

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("\nMetrics Results")
    print("abs_rel: {}".format(abs_rel.mean()))
    print("sq_rel: {}".format(sq_rel.mean()))
    print("rms_rel: {}".format(rms.mean()))
    print("log_rms: {}".format(log_rms.mean()))
    print("a1: {}".format(a1.mean()))
    print("a2: {}".format(a2.mean()))
    print("a3: {}".format(a3.mean()))

    print("Testing completed.")


# ===== #
#  Main #
# ===== #
if __name__ == '__main__':
    args = args_handler()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise SystemError

    print("Done.")
