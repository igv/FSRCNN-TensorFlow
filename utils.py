"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
from math import ceil
import subprocess
import io
from random import randrange, shuffle

import tensorflow.compat.v1 as tf
from PIL import Image
import numpy as np
import multiprocessing

FLAGS = tf.app.flags.FLAGS

downsample = True

def preprocess(path, scale=3, distort=False):
  """
  Preprocess single image file
    (1) Read original image
    (2) Downsample by scale factor
    (3) Normalize
  """
  try:
    from wand.image import Image
  except:
    from PIL import Image
    image = Image.open(path).convert('L')
    (width, height) = image.size

    if downsample:
        image = image.crop((0, 0, width - width % scale, height - height % scale))

        (width, height) = image.size
        label_ = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape((height, width))

        (new_width, new_height) = width // scale, height // scale
        scaled_image = image.resize((new_width, new_height), Image.BICUBIC)
        image.close()

        if distort==True and randrange(3) == 0:
            buf = io.BytesIO()
            scaled_image.convert('RGB').save(buf, "JPEG", quality=randrange(85, 95, 5))
            buf.seek(0)
            scaled_image = Image.open(buf).convert('L')
            #scaled_image.convert('RGB').save("lowres.png")
            #subprocess.call(['ffmpeg', '-y', '-i', 'lowres.png', '-c:v', 'libx264', '-crf', '20', 'lowres.mkv'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #subprocess.call(['ffmpeg', '-y', '-i', 'lowres.mkv', '-vframes', '1', 'lowres.png'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #scaled_image = Image.open('lowres.png').convert('L')

        input_ = np.frombuffer(scaled_image.tobytes(), dtype=np.uint8).reshape((new_height, new_width))
    else:
        input_ = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(height, width)
        scaled_image = image.resize((width * scale, height * scale), Image.BICUBIC)
        (width, height) = scaled_image.size
        label_ = np.frombuffer(scaled_image.tobytes(), dtype=np.uint8).reshape(height, width)
  else:
    with Image(filename=path) as img:
        img.alpha_channel = False
        img.transform_colorspace("ycbcr")
        if downsample:
            img.crop(width = img.width - img.width % scale, height = img.height - img.height % scale)
            label_ = np.frombuffer(img.make_blob('YCbCr'), dtype=np.uint8).reshape(img.height, img.width, 3)[:,:,0]
            img.resize(width = img.width // scale, height = img.height // scale, filter = "lanczos2", blur=1.0)
            if distort==True and randrange(3) == 0:
                img.compression_quality = randrange(85, 95, 5)
                img.transform_colorspace("rgb")
                jpeg_bin = img.make_blob('jpeg')
                img = Image(blob=jpeg_bin)
            input_ = np.frombuffer(img.make_blob('YCbCr'), dtype=np.uint8).reshape(img.height, img.width, 3)[:,:,0]
        else:
            input_ = np.frombuffer(img.make_blob('YCbCr'), dtype=np.uint8).reshape(img.height, img.width, 3)[:,:,0]
            img.resize(width = img.width * scale, height = img.height * scale, filter = "catrom")
            label_ = np.frombuffer(img.make_blob('YCbCr'), dtype=np.uint8).reshape(img.height, img.width, 3)[:,:,0]

  return input_ / 255, label_ / 255

def prepare_data(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.train:
    data_dir = os.path.join(os.getcwd(), dataset)
    data = []
    for files in ('*.bmp', '*.png'):
        data.extend(glob.glob(os.path.join(data_dir, files)))
    shuffle(data)
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))

  return data

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def train_input_worker(args):
  image_data, config = args
  image_size, label_size, stride, scale, padding, distort = config

  single_input_sequence, single_label_sequence = [], []

  input_, label_ = preprocess(image_data, scale, distort=distort)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size + 1, stride):
    for y in range(0, w - image_size + 1, stride):
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [single_input_sequence, single_label_sequence]

def multiprocess_train_setup(config):
  """
  Spawns several processes to pre-process the data
  """
  if downsample == False:
    import sys
    sys.exit()

  data = prepare_data(config.sess, dataset=config.data_dir)

  with multiprocessing.Pool(max(multiprocessing.cpu_count() - 1, 1)) as pool:
    config_values = [config.image_size, config.label_size, config.stride, config.scale, config.padding // 2, config.distort]
    results = pool.map(train_input_worker, [(data[i], config_values) for i in range(len(data))])

  sub_input_sequence, sub_label_sequence = [], []

  for image in range(len(results)):
    single_input_sequence, single_label_sequence = results[image]
    sub_input_sequence.extend(single_input_sequence)
    sub_label_sequence.extend(single_label_sequence)

  arrdata = np.asarray(sub_input_sequence)
  arrlabel = np.asarray(sub_label_sequence)

  return (arrdata, arrlabel)

def test_input_setup(config):
  sess = config.sess

  # Load data path
  data = prepare_data(sess, dataset="Test")

  input_, label_ = preprocess(data[2], config.scale)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  arrdata = np.pad(input_.reshape([1, h, w, 1]), ((0,0),(2,2),(2,2),(0,0)), 'reflect')

  if len(label_.shape) == 3:
    h, w, _ = label_.shape
  else:
    h, w = label_.shape

  arrlabel = label_.reshape([1, h, w, 1])

  return (arrdata, arrlabel)

def merge(config, Y):
  """
  Merges super-resolved image with chroma components
  """
  h, w = Y.shape[1], Y.shape[2]
  Y = Y.reshape(h, w, 1) * 255
  Y = Y.round().astype(np.uint8)

  data = prepare_data(config.sess, dataset="Test")
  src = Image.open(data[2]).convert('YCbCr')
  (width, height) = src.size
  if downsample is False:
    src = src.resize((width * config.scale, height * config.scale), Image.BICUBIC)
    (width, height) = src.size
  CbCr = np.frombuffer(src.tobytes(), dtype=np.uint8).reshape(height, width, 3)[:,:,1:]

  img = np.concatenate((Y, CbCr), axis=-1)

  return img

def save_params(sess, params):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  h = open(param_dir + "weights{}.txt".format('_'.join(str(i) for i in params)), 'w')

  variables = dict((var.name, sess.run(var)) for var in tf.trainable_variables())

  for name, weights in variables.items():
    h.write("{} =\n".format(name[:name.index(':')]))

    if len(weights.shape) < 4:
        h.write("{}\n\n".format(weights.flatten().tolist()))
    else:
        h.write("[")
        sep = False
        for filter_x in range(len(weights)):
          for filter_y in range(len(weights[filter_x])):
            filter_weights = weights[filter_x][filter_y]
            for input_channel in range(len(filter_weights)):
              for output_channel in range(len(filter_weights[input_channel])):
                val = filter_weights[input_channel][output_channel]
                if sep:
                    h.write(', ')
                h.write("{}".format(val))
                sep = True
              h.write("\n  ")
        h.write("]\n\n")

  h.close()

def array_image_save(array, image_path):
  """
  Converts np array to image and saves it
  """
  image = Image.fromarray(array, 'YCbCr')
  if image.mode != 'RGB':
    image = image.convert('RGB')
  image.save(image_path)
  print("Saved image: {}".format(image_path))

def gradient_sensitive_loss(img1, img2):
    dY = tf.image.sobel_edges(img1) / 4.
    dX = tf.image.sobel_edges(img2) / 4.
    M = tf.sqrt(tf.square(dY[:,:,:,:,0]) + tf.square(dY[:,:,:,:,1]))
    #dY = tf.image.sobel_edges(img1 * M)
    #dX = tf.image.sobel_edges(img2 * M)
    return tf.losses.absolute_difference(dY, dX) \
         + tf.losses.absolute_difference((1.0 - M) * img1, (1.0 - M) * img2, weights=2.0)

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=7):
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    img1 = tf.clip_by_value(img1, 0, L)
    img2 = tf.clip_by_value(img2, 0, L)
    window = _tf_fspecial_gauss(size, 1.5)
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID', data_format='NHWC') - mu1_mu2

    value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    if cs_map == False:
        value *= tf.pow((2.0 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1), 55.0)

    return tf.reduce_mean(value) if mean_metric else value

def tf_ms_ssim(img1, img2, weights=[0.2, 0.8], size=3):
    weights = weights / np.sum(weights)
    window = _tf_fspecial_gauss(5, 1.08)
    mssim = []
    for i in range(len(weights)):
        mssim.append(tf_ssim(img1, img2, i < len(weights) - 1, size=size))
        img1 = tf.pad(img1, [[0,0],[2,2],[2,2],[0,0]], 'reflect')
        img2 = tf.pad(img2, [[0,0],[2,2],[2,2],[0,0]], 'reflect')
        img1 = tf.nn.conv2d(img1, window, [1,2,2,1], 'VALID')
        img2 = tf.nn.conv2d(img2, window, [1,2,2,1], 'VALID')

    value = tf.reduce_sum(tf.multiply(tf.stack(mssim), weights))

    return value

def bilinear_upsample_weights(factor, channels):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    filter_size = 2 * factor - factor % 2
    center = factor - (1 if factor % 2 == 1 else 0.5)
    og = np.ogrid[:filter_size, :filter_size]
    upsample_kernel = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weights = np.zeros((filter_size, filter_size, channels, channels), dtype=np.float32)
    for i in range(channels):
        weights[:, :, i, i] = upsample_kernel
    return weights

def bicubic_kernel(x, B=1/3., C=1/3.):
    """https://de.wikipedia.org/wiki/Mitchell-Netravali-Filter"""
    if abs(x) < 1:
      return 1/6. * ((12-9*B-6*C)*abs(x)**3 + ((-18+12*B+6*C)*abs(x)**2 + (6-2*B)))
    elif 1 <= abs(x) and abs(x) < 2:
      return 1/6. * ((-B-6*C)*abs(x)**3 + (6*B+30*C)*abs(x)**2 + (-12*B-48*C)*abs(x) + (8*B+24*C))
    else:
      return 0

def build_filter(factor, B, C, channels=1):
    size = factor * 4
    k = np.zeros((size), dtype=np.float32)
    for i in range(size):
        x = (1 / factor) * (i - np.floor(size / 2) + 0.5)
        k[i] = bicubic_kernel(x, B, C)
    k = k / np.sum(k)
    k = np.outer(k, k)
    weights = np.zeros((size, size, channels, channels), dtype=np.float32)
    for i in range(channels):
        weights[:, :, i, i] = k
    return weights

def bicubic_downsample(x, factor, B=1/3., C=1/3.):
    """Downsample x by a factor of factor, using the filter built by build_filter()
    x: a rank 4 tensor with format NHWC
    factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
    """
    # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
    kernel_size = factor * 4
    padding = kernel_size - factor
    pad_top = padding // 2
    pad_bottom = padding - pad_top
    pad_left = padding // 2
    pad_right = padding - pad_left
    # apply mirror padding
    x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
    # downsampling performed by strided conv
    x = tf.nn.conv2d(x, build_filter(factor, B, C), [1,factor,factor,1], 'VALID', data_format='NHWC')
    return x
