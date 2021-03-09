import os
import io
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import time

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

from lxml import etree
import xml.etree.ElementTree as ET

# Load model
ROOT = '/home/JulioCesar/flores/MIA2/landmark_detection'
PATH_TO_SAVED_MODEL = os.path.join(ROOT, 'exported', 'train_2', 'ckpt-6', 'saved_model')
PATH_TO_LABELS = os.path.join(ROOT, 'landmarks_label_map.pbtxt')

print('Model path {}'.format(PATH_TO_SAVED_MODEL))
print('Label path {}'.format(PATH_TO_LABELS))

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Model loaded')

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
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

ROOT_IMGS = '/home/JulioCesar/flores/MIA2/videos_data'
IMAGES_FOLDER = os.path.join(ROOT_IMGS, 'frames', 'DJI_0289_alto')
TARGET_ANNOTATIONS = os.path.join(ROOT, 'Inference_Annotations_1')
if not os.path.exists(TARGET_ANNOTATIONS):
  os.mkdir(TARGET_ANNOTATIONS)
images_list = os.listdir(IMAGES_FOLDER)
num_images = len(images_list)
print('Images folder {}'.format(IMAGES_FOLDER))
print('Annotations folder {}'.format(TARGET_ANNOTATIONS))

for idx in range(190, num_images):
  print('{}/{}. Running inference'.format(idx + 1, len(images_list)))
  img_id = 'img_{}'.format(idx)
  image_name = img_id + '.jpg'
  image_path = os.path.join(IMAGES_FOLDER, image_name)
  image_np = load_image_into_numpy_array(image_path)
  height, width, channels = image_np.shape
  input_tensor = tf.convert_to_tensor(image_np)
  input_tensor = input_tensor[tf.newaxis, ...]

  # run detections
  detections = detect_fn(input_tensor)

  num_detections = int(detections.pop('num_detections'))

  detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
  detections['num_detections'] = num_detections
  
  # detection_classes should be ints
  detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

  # create annotation
  # header
  annotation = ET.Element('annotation')
  ET.SubElement(annotation, 'folder').text = 'Images'
  ET.SubElement(annotation, 'filename').text = image_name
  ET.SubElement(annotation, 'path').text = image_path
  source = ET.SubElement(annotation, 'source')
  ET.SubElement(source, 'database').text = 'Unknown'
  size = ET.SubElement(annotation, 'size')
  ET.SubElement(size, 'width').text = str(width)
  ET.SubElement(size, 'height').text = str(height)
  ET.SubElement(size, 'depth').text = str(channels)
  ET.SubElement(annotation, 'segmented').text = '0'

  min_score_thresh = 0.5
  for i in range(len(detections['detection_scores'])):
    if detections['detection_scores'][i] > min_score_thresh:
      object_name = category_index[detections['detection_classes'][i]]['name']
      ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
      xmin = int(xmin * width)
      xmax = int(xmax * width)
      ymin = int(ymin * height)
      ymax = int(ymax * height)

      object_annotation = ET.SubElement(annotation, 'object')
      ET.SubElement(object_annotation, 'name').text = object_name
      ET.SubElement(object_annotation, 'pose').text = 'Unspecified'
      ET.SubElement(object_annotation, 'truncated').text = '0'
      ET.SubElement(object_annotation, 'difficult').text = '0'
      bndbox = ET.SubElement(object_annotation, 'bndbox')
      ET.SubElement(bndbox, 'xmin').text = str(xmin)
      ET.SubElement(bndbox, 'ymin').text = str(ymin)
      ET.SubElement(bndbox, 'xmax').text = str(xmax)
      ET.SubElement(bndbox, 'ymax').text = str(ymax)
  
  tree = ET.ElementTree(annotation)
  tree_root = tree.getroot()
  xmlstr = ET.tostring(tree_root, encoding='utf8', method='xml')

  tree.write(os.path.join(TARGET_ANNOTATIONS, img_id + '.xml'))
  # break