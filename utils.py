"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])
    else:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

#
import tensorflow as tf
import glob
import os

#
CROP_SIZE  = 256
SCALE_SIZE = 286

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

#
def load_batch_examples(dataset_name, which_direction, batch_size):
    # input directory
    input_dir = './datasets/{}/train'.format(dataset_name)
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    # input files
    input_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        # Make a Dataset of file names including all the JPEG images files in
        # the relative image directory.
        dataset = tf.data.Dataset.from_tensor_slices(input_paths)

        # Make a Dataset of image tensors by reading and decoding the files.
        dataset = dataset.map(lambda x: load_one_example(decode, x, which_direction), num_parallel_calls=None)
    #
    dataset = dataset.map(transform_pairs, num_parallel_calls=None).repeat().batch(batch_size)

    return dataset, len(input_paths)

def load_one_example(decode, filename, which_direction):
    # Make a Dataset of image tensors by reading and decoding the files.
    raw_input = decode(tf.read_file(filename))
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)

    raw_input.set_shape([None, None, 3])

    # break apart image pair and move to range [-1, 1]
    width = tf.shape(raw_input)[1] # [height, width, channels]
    a_images = preprocess(raw_input[:,:width//2,:])
    b_images = preprocess(raw_input[:,width//2:,:])

    #
    if which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    return inputs, targets

def transform_pairs(inputs, targets, flip=False, scale_size=SCALE_SIZE, crop_size=CROP_SIZE):
    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image, flip=flip, scale_size=scale_size, crop_size=crop_size):
        r = image
        if flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1, seed=seed)), dtype=tf.int32)
        if scale_size > crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], crop_size, crop_size)
        elif scale_size < crop_size:
            raise Exception("scale size cannot be less than crop size")

        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    return input_images, target_images

if __name__ == '__main__':
    # dataset
    dataset_name = 'facades'
    image_dataset, steps_per_epoch = load_batch_examples(dataset_name, which_direction="AtoB", batch_size=1)

    #
    iter = image_dataset.make_one_shot_iterator()
    inputs_batch, targets_batch = iter.get_next()

    # for testing
    inputs_batch = deprocess(inputs_batch[0])
    targets_batch = deprocess(targets_batch[0])

    with tf.Session() as sess:
        img_A, img_B = sess.run([inputs_batch, targets_batch])

    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.imshow(img_A)
    plt.subplot(212)
    plt.imshow(img_B)

    plt.show()


