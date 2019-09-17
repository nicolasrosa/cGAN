#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Helpful Links
# http://kieleth.blogspot.com.br/2014/03/opencv-calculate-average-fps-in-python.html

# ===========
#  Libraries
# ===========
import argparse
import sys

import cv2
import numpy as np
import imageio
# from dataset_handler import load_and_scale_image
from model import cGAN

def load_and_scale_image(image_input, size=(256, 256)): # TODO: Modificar funcão no dataset_handler, para lidar tanto com filepath ou image_input
    # image_input = imageio.imread(filepath)
    image_input = cv2.resize(image_input, size, interpolation=cv2.INTER_AREA)
    image_input = np.expand_dims(image_input, axis=0)
    # image_input = image_input.astype(np.float32)

    return (image_input / 127.5) - 1

# ==================
#  Global Variables
# ==================
SAVE_IMAGES = False

global max_depth
global timer


# ===========
#  Functions
# ===========
def argument_handler():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=str, help="Select which gpu to run the code", default='0')
    parser.add_argument('-r', '--model_path', help='Converted parameters for the model', default='')
    parser.add_argument('-i', '--video_path', help='Directory of images to predict', required=True)
    parser.add_argument("-n", "--model_name", type=str, help="Chooses the network", required=True)

    return parser.parse_args()


def circular_counter(max_value):
    """helper function that creates an eternal counter till a max value"""
    x = 0
    while True:
        if x == max_value:
            x = 0
        x += 1
        yield x


class maxDepth:
    def __init__(self):
        self.counter_len = 1000
        self.l_fps_history = [10 for _ in range(self.counter_len)]
        self.max_depth_counter = circular_counter(self.counter_len)

    def update(self, src):
        max_depth = np.max(src)
        self.l_fps_history[next(self.max_depth_counter) - 1] = max_depth
        return max_depth

    @property
    def get_avg(self):
        return sum(self.l_fps_history) / float(self.counter_len)


class CvTimer:
    def __init__(self):
        self.tick_frequency = cv2.getTickFrequency()
        self.tick_at_init = cv2.getTickCount()
        self.last_tick = self.tick_at_init
        self.counter_len = 100
        self.l_fps_history = [10 for _ in range(self.counter_len)]
        self.fps_counter = circular_counter(self.counter_len)

    def reset(self):
        self.last_tick = cv2.getTickCount()

    @staticmethod
    def get_tick_now():
        return cv2.getTickCount()

    @property
    def fps(self):
        fps = self.tick_frequency / (self.get_tick_now() - self.last_tick)
        self.l_fps_history[next(self.fps_counter) - 1] = fps
        return fps

    @property
    def get_avg(self):
        return sum(self.l_fps_history) / float(self.counter_len)


def apply_overlay(frame, pred_jet_resized):
    alpha = 0.5
    background = frame.copy()
    overlay = pred_jet_resized.copy()
    overlay = cv2.addWeighted(background, alpha, overlay, 1 - alpha, 0)

    return overlay


def convertScaleAbs(src):
    global max_depth

    # print(max_depth.fps(src), max_depth.get_avg)
    max_depth.update(src)

    # return cv2.convertScaleAbs(src * (255 / np.max(src)))
    return cv2.convertScaleAbs(src * (255 / max_depth.get_avg))


def generate_colorbar(height, colormap, inv=False):
    colorbar = np.zeros(shape=(height, 45), dtype=np.uint8)

    for row in range(colorbar.shape[0]):
        for col in range(colorbar.shape[1]):
            # print(row, col)
            colorbar[row, col] = int((row / colorbar.shape[0]) * 255)

    if inv:
        colorbar = 255 - colorbar

    colorbar = cv2.applyColorMap(colorbar, colormap)

    cv2.putText(colorbar, "%0.2f-" % max_depth.get_avg, (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255))
    cv2.putText(colorbar, "%0.2f-" % (max_depth.get_avg / 2), (1, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255))
    cv2.putText(colorbar, "%0.2f-" % 0.0, (10, height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

    return colorbar


def process_images(frame, pred, remove_sky=False):
    global timer
    suffix = ''

    if remove_sky:
        # Remove Sky
        crop_height_perc = 0.3
        frame = frame[int(crop_height_perc * frame.shape[0]):, :, :]
        pred = pred[:, int(crop_height_perc * pred.shape[1]):, :, :]

        # print(frame.shape)
        # print(pred.shape)

        suffix = ' (without sky)'

    # Change Data Scale from meters to uint8
    pred_uint8 = cv2.convertScaleAbs(pred[0])
    pred_scaled_uint8 = convertScaleAbs(pred[0])

    # Apply Median Filter
    pred_median = cv2.medianBlur(pred[0], 3)
    pred_median_scaled_uint8 = convertScaleAbs(pred_median)

    # Change Colormap
    pred_jet = cv2.applyColorMap(255 - pred_median_scaled_uint8, cv2.COLORMAP_JET)
    pred_hsv = cv2.applyColorMap(pred_median_scaled_uint8, cv2.COLORMAP_HSV)

    # Resize
    pred_jet_resized = cv2.resize(pred_jet, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
    pred_hsv_resized = cv2.resize(pred_hsv, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

    # Apply the overlay
    overlay = apply_overlay(frame, pred_jet_resized)

    # Write text on Image
    cv2.putText(frame, "fps=%0.2f avg=%0.2f" % (timer.fps, timer.get_avg), (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))

    # Generate Colorbar
    colorbar_jet = generate_colorbar(height=pred_jet_resized.shape[0], colormap=cv2.COLORMAP_JET)
    colorbar_hsv = generate_colorbar(height=pred_jet_resized.shape[0], colormap=cv2.COLORMAP_HSV, inv=True)

    # Concatenates Images
    conc = cv2.hconcat([pred_uint8, pred_scaled_uint8, pred_median_scaled_uint8])
    conc2 = cv2.hconcat([frame, pred_jet_resized, colorbar_jet, pred_hsv_resized, colorbar_hsv, overlay])

    # Debug
    if args.debug:
        print(pred.shape, pred.dtype)
        print(pred_uint8.shape, pred_uint8.dtype)
        print(pred_jet_resized.shape, pred_jet_resized.dtype)
        print(pred_hsv_resized.shape, pred_hsv_resized.dtype)
        print(overlay.shape, overlay.dtype)

    # Display the resulting frame - OpenCV
    # cv2.imshow('frame', frame)
    # cv2.imshow('pred', pred_uint8)
    # cv2.imshow('pred_jet (scaled, median, resized)', pred_jet_resized)
    # cv2.imshow('pred(scaled)', pred_scaled_uint8)
    # cv2.imshow('pred_hsv (scaled, median, resized)', pred_hsv_resized)
    # cv2.imshow('pred(scaled, median)', pred_median_scaled_uint8)
    # cv2.imshow('overlay', overlay)
    cv2.imshow('pred, pred(scaled), pred(scaled, median)' + suffix, conc)
    cv2.imshow('frame, pred_jet, pred_hsv, overlay' + suffix, conc2)


timer = CvTimer()
max_depth = maxDepth()
args = argument_handler()


# ======
#  Main
# ======
def main():
    print(args.model_path)
    print(args.video_path)

    # Load from Camera or Video
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(args.video_path)

    if not cap.isOpened():  # Check if it succeeded
        print("It wasn't possible to open the camera.")
        return -1

    # ----------------
    #  Building Graph
    # ----------------
    # Defines Input shape
    img_rows, img_cols, channels, channels_depth = 256, 256, 3, 1

    img_shape = (img_rows, img_cols, channels)
    depth_shape = (img_rows, img_cols, channels_depth)

    batch_size = 1

    # --------
    # Model
    # --------
    model = cGAN(img_shape, depth_shape)
    generator = model.select_generator_model(args.model_name)
    generator.summary()

    # -----------------------
    # Load generator weights
    # -----------------------
    # TODO: Fazer aquele rotina de detectar quais modelos estão disponíveis. Dependendo do nome, selecionar arquitetura de rede corretamente, para então logar os pesos.
    print('\nLoading the model...')
    # generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/kitti_morphological/2019-09-02_10-29-44/weights_generator_bce.h5')
    # generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/weights_generator_linear4.h5')
    # generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/resnet/2019-09-13_11-03-36/weights_generator_bce.h5')
    generator.load_weights('/home/nicolas/MEGA/workspace/cGAN/output/resnet_raul/weights_generator_resnet.h5')

    count = 0
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.resize(frame, (img_cols, img_rows), interpolation=cv2.INTER_AREA)

        # Evaluate the network for the given image
        try:
            imgs_B = load_and_scale_image(frame)
            fake_A = generator.predict(imgs_B)
            # pred, pred_50, pred_80 = sess.run([tf_pred, tf_pred_50, tf_pred_80], feed_dict={input_node: frame})
            # pred_50_uint8_scaled = convertScaleAbs(pred_50[0])
            # pred_80_uint8_scaled = convertScaleAbs(pred_80[0])
            # cv2.imshow('pred_50 (scaled)', pred_50_uint8_scaled)
            # cv2.imshow('pred_80 (scaled)', pred_80_uint8_scaled)
        except UnboundLocalError:
            # pred = sess.run(tf_pred, feed_dict={input_node: frame})
            pass

        # Debug
        if args.debug:
            print(frame)
            print(frame.shape, frame.dtype)
            print()
            print(fake_A)
            print(fake_A.shape, fake_A.dtype)
            input("Continue...")

        # ------------------ #
        #  Image Processing  #
        # ------------------ #
        # process_images(frame, pred) # FIXME: Para redes que não consideram o ceu, os valores da predição sujam o valor de max_depth
        process_images(frame, fake_A, remove_sky=True)

        # # Save Images
        # if SAVE_IMAGES:
        #     cv2.imwrite(settings.output_dir + "fcrn_cv/frame%06d.png" % count, frame)
        #     cv2.imwrite(settings.output_dir + "fcrn_cv/pred%06d.png" % count, pred_uint8)
        #     cv2.imwrite(settings.output_dir + "fcrn_cv/jet%06d.png" % count, pred_jet_resized)
        #     count += 1

        timer.reset()

        if cv2.waitKey(1) & 0xFF == ord('q'):  # without waitKey() the images are not shown.
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    print("Done.")
    sys.exit()


if __name__ == '__main__':
    main()
