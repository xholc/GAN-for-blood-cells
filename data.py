import cv2
import numpy as np
import tensorflow as tf
from utililies import get_bit_plane


IMG_WIDTH = 256
IMG_HEIGHT = 256


def load_numpy(image_file, mode_flag):
    image = tf.io.read_file(image_file)
    image_r = tf.image.decode_jpeg(image).numpy()
    r, g, b = image_r[:, :, 0], image_r[:, :, 1], image_r[:, :, 2]
    
    if mode_flag == 0:  # train mode
        # this parameter comes from  tried out on  training set
        rb, gb, bb = 4, 4, 5
    else:  # test mode
        # this parameter comes from  tried out on  testingg set
        rb, gb, bb = 6, 6, 5

    bit_plane_r = get_bit_plane(r, rb)
    kr = (bit_plane_r + 1) * (2**rb) - 1
    bit_plane_g = get_bit_plane(g, gb)
    kg = (bit_plane_g + 1) * (2**gb) - 1
    bit_plane_b = get_bit_plane(b, bb)
    kb = (bit_plane_b + 1) * (2**bb) - 1

    result_fig = np.stack((kr, kg, kb), axis=-1)

    return result_fig.astype(np.float32), image_r.astype(np.float32)

def load(image_file, mode):
    mode_flag = tf.constant(0 if mode == 'train' else 1, dtype=tf.int32)
    # tf.numpy_function(numpy_function: input should be tensor matched with inp, [inp: Tensor], [tar:type specified Tensor]),
    # if using decorator you should not secicify numpy_function & inp
    input_image, real_image = tf.numpy_function(load_numpy, [image_file, mode_flag], [tf.float32, tf.float32])
    input_image.set_shape([None, None, 3])
    real_image.set_shape([None, None, 3])

    return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # 調整大小為286 x 286 x 3
    input_image, real_image = resize(input_image, real_image, 286, 286)

    # 隨機裁減為256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
    # 隨機水平翻轉
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

def load_image_train(image_file):
    input_image, real_image = load(image_file, mode='train')
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

def load_image_test(image_file):
    input_image, real_image = load(image_file, mode='test')
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image