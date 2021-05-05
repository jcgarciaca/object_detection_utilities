# Imports and setup
import os
import io
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time
import tensorflow as tf

from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

# Load model
ROOT = '/home/JulioCesar/flores/MIA2/landmark_detection/marcas_2'
PATH_TO_SAVED_MODEL = os.path.join(ROOT, 'exported', 'train_2', 'ckpt-5', 'saved_model')
PATH_TO_LABELS = os.path.join(ROOT, 'landmarks_label_map.pbtxt')

print('Model path {}'.format(PATH_TO_SAVED_MODEL))
print('Label path {}'.format(PATH_TO_LABELS))

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Model loaded')

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)
print('Category index', category_index)


# Run detections
def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  return np.array(Image.open(path))

IMAGES_FOLDER = os.path.join(ROOT, 'frames', 'DJI_0025')
TARGET_FOLDER = os.path.join(ROOT, 'frames', 'Processed')

if not os.path.exists(TARGET_FOLDER):
  os.makedirs(TARGET_FOLDER)

images_list = os.listdir(IMAGES_FOLDER)
print('Images folder {}'.format(IMAGES_FOLDER))
print('Target folder {}'.format(TARGET_FOLDER))

min_score_thresh = 0.5
num_images = len(images_list)
clusters_lst = []
print('Found {} images'.format(num_images))

for idx in range(num_images):
  image_name = 'img_{}.jpg'.format(idx)
  print('{}/{}. Running inference for {}'.format(idx + 1, len(images_list), image_name))
  image_np = load_image_into_numpy_array(os.path.join(IMAGES_FOLDER, image_name))
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  # run detections
  detections = detect_fn(input_tensor)

  num_detections = int(detections.pop('num_detections'))

  detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
  detections['num_detections'] = num_detections
  
  # detection_classes should be ints
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
  
  image_np_with_detections = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=20,
      min_score_thresh=min_score_thresh,
      agnostic_mode=False)

  h, w, channels = image_np.shape
  y_ = []

  for idx_, score in enumerate(detections['detection_scores']):
    if score >= min_score_thresh:
      ymin, xmin, ymax, xmax = detections['detection_boxes'][idx_]
      ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
      y_.append((ymin + ymax) / 2)

  # only one object detected and it is in central area of image
  lower_limit = h / 4
  upper_limit = (3 * h) / 4
  if len(y_) == 1 and y_[0] > lower_limit and y_[0] < upper_limit:
    extracted_np = image_np_with_detections.copy()
  else:
    # create black image
    extracted_np = np.zeros((h, w, channels))

  imgs = [Image.fromarray(image_np_with_detections, 'RGB'), Image.fromarray(extracted_np, 'RGB')]

  concat_im = Image.new('RGB', (2 * w, h))
  x_offset = 0
  for im in imgs:
    concat_im.paste(im, (x_offset, 0))
    x_offset += im.size[0]
  
  concat_im.save(os.path.join(TARGET_FOLDER, 'img_{}.jpg'.format(idx)))