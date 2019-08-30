import os
import time

import cv2
import imageio
import numpy as np
from keras_preprocessing.image import img_to_array, load_img


def load_dataset(dataset_name):
    train_images, train_labels = None, None

    if not (
            os.path.exists(
                'data/{}_train.txt'.format(dataset_name))):  # and os.path.exists('kitti_guidenet_test(2).txt')):
        timer1 = -time.time()

        bad_words = ['image_03',
                     '2011_09_28_drive_0053_sync',
                     '2011_09_28_drive_0054_sync',
                     '2011_09_28_drive_0057_sync',
                     '2011_09_28_drive_0065_sync',
                     '2011_09_28_drive_0066_sync',
                     '2011_09_28_drive_0068_sync',
                     '2011_09_28_drive_0070_sync',
                     '2011_09_28_drive_0071_sync',
                     '2011_09_28_drive_0075_sync',
                     '2011_09_28_drive_0077_sync',
                     '2011_09_28_drive_0078_sync',
                     '2011_09_28_drive_0080_sync',
                     '2011_09_28_drive_0082_sync',
                     '2011_09_28_drive_0086_sync',
                     '2011_09_28_drive_0087_sync',
                     '2011_09_28_drive_0089_sync',
                     '2011_09_28_drive_0090_sync',
                     '2011_09_28_drive_0094_sync',
                     '2011_09_28_drive_0095_sync',
                     '2011_09_28_drive_0096_sync',
                     '2011_09_28_drive_0098_sync',
                     '2011_09_28_drive_0100_sync',
                     '2011_09_28_drive_0102_sync',
                     '2011_09_28_drive_0103_sync',
                     '2011_09_28_drive_0104_sync',
                     '2011_09_28_drive_0106_sync',
                     '2011_09_28_drive_0108_sync',
                     '2011_09_28_drive_0110_sync',
                     '2011_09_28_drive_0113_sync',
                     '2011_09_28_drive_0117_sync',
                     '2011_09_28_drive_0119_sync',
                     '2011_09_28_drive_0121_sync',
                     '2011_09_28_drive_0122_sync',
                     '2011_09_28_drive_0125_sync',
                     '2011_09_28_drive_0126_sync',
                     '2011_09_28_drive_0128_sync',
                     '2011_09_28_drive_0132_sync',
                     '2011_09_28_drive_0134_sync',
                     '2011_09_28_drive_0135_sync',
                     '2011_09_28_drive_0136_sync',
                     '2011_09_28_drive_0138_sync',
                     '2011_09_28_drive_0141_sync',
                     '2011_09_28_drive_0143_sync',
                     '2011_09_28_drive_0145_sync',
                     '2011_09_28_drive_0146_sync',
                     '2011_09_28_drive_0149_sync',
                     '2011_09_28_drive_0153_sync',
                     '2011_09_28_drive_0154_sync',
                     '2011_09_28_drive_0155_sync',
                     '2011_09_28_drive_0156_sync',
                     '2011_09_28_drive_0160_sync',
                     '2011_09_28_drive_0161_sync',
                     '2011_09_28_drive_0162_sync',
                     '2011_09_28_drive_0165_sync',
                     '2011_09_28_drive_0166_sync',
                     '2011_09_28_drive_0167_sync',
                     '2011_09_28_drive_0168_sync',
                     '2011_09_28_drive_0171_sync',
                     '2011_09_28_drive_0174_sync',
                     '2011_09_28_drive_0177_sync',
                     '2011_09_28_drive_0179_sync',
                     '2011_09_28_drive_0183_sync',
                     '2011_09_28_drive_0184_sync',
                     '2011_09_28_drive_0185_sync',
                     '2011_09_28_drive_0186_sync',
                     '2011_09_28_drive_0187_sync',
                     '2011_09_28_drive_0191_sync',
                     '2011_09_28_drive_0192_sync',
                     '2011_09_28_drive_0195_sync',
                     '2011_09_28_drive_0198_sync',
                     '2011_09_28_drive_0199_sync',
                     '2011_09_28_drive_0201_sync',
                     '2011_09_28_drive_0204_sync',
                     '2011_09_28_drive_0205_sync',
                     '2011_09_28_drive_0208_sync',
                     '2011_09_28_drive_0209_sync',
                     '2011_09_28_drive_0214_sync',
                     '2011_09_28_drive_0216_sync',
                     '2011_09_28_drive_0220_sync',
                     '2011_09_28_drive_0222_sync']

        with open('{}_train.txt'.format(dataset_name)) as oldfile, open('{}_train(2).txt'.format(dataset_name),
                                                                        'w') as newfile:
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    newfile.write(line)

        timer1 += time.time()

    else:

        timer1 = -time.time()

        try:

            def read_text_file(filename, dataset_path):
                print("\n[Dataloader] Loading '%s'..." % filename)
                try:
                    data = np.genfromtxt(filename, dtype='str', delimiter='\t')
                    # print(data.shape)

                    # Parsing Data
                    image_filenames = list(data[:, 0])
                    depth_filenames = list(data[:, 1])

                    timer = -time.time()
                    image_filenames = [dataset_path + filename for filename in image_filenames]
                    depth_filenames = [dataset_path + filename for filename in depth_filenames]
                    timer += time.time()
                    print('time:', timer, 's\n')

                except OSError:
                    raise OSError("Could not find the '%s' file." % filename)

                return image_filenames, depth_filenames

            image_filenames, depth_filenames = read_text_file(
                filename='/home/nicolas/MEGA/workspace/cGAN/data/{}_train.txt'.format(dataset_name),
                dataset_path='/media/nicolas/nicolas_seagate/datasets/kitti/')

            image = sorted(image_filenames)
            depth = sorted(depth_filenames)
            # image_val = sorted(image_validation)
            # depth_val = sorted(depth_validation)

            train_images = image
            train_labels = depth
            # test_images = image_val
            # test_labels = depth_val

            print(len(image))
            print(len(depth))

            timer1 += time.time()

        except OSError:
            raise SystemExit

    return train_images, train_labels


def load_and_scale_image(filepath):
    image_input = imageio.imread(filepath)
    image_input = cv2.resize(image_input, (256, 256), interpolation=cv2.INTER_AREA)
    image_input = np.expand_dims(image_input, axis=0)
    # image_input = image_input.astype(np.float32)  # float64 -> float32

    return (image_input / 127.5) - 1


def load_and_scale_depth(filepath):
    image_input = imageio.imread(filepath)
    image_input = cv2.resize(image_input, (256, 256), interpolation=cv2.INTER_AREA)
    image_input = np.expand_dims(image_input, axis=-1) / 256.0  # TODO: Nem todos datasets serão 256.0
    # image_input = image_input.astype(np.float32)  # float64 -> float32  # This will increase precisions errors from 1e-14 to 1e-4!
    image_input = np.expand_dims(image_input, axis=0)

    # print(image_input.shape, image_input.dtype)
    # print(np.min(image_input), np.max(image_input))
    # input('aki')

    return np.log(image_input+1)
    # return (image_input / 42.5) - 1
