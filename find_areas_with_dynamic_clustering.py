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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import tensorflow as tf

from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils

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

ROOT_IMGS = '/home/JulioCesar/flores/MIA2/videos_data'
IMAGES_FOLDER = os.path.join(ROOT_IMGS, 'frames', 'DJI_0289_alto')
TARGET_FOLDER = os.path.join(ROOT, 'Y_Areas_Dyn_2')
if not os.path.exists(TARGET_FOLDER):
  os.mkdir(TARGET_FOLDER)
images_list = os.listdir(IMAGES_FOLDER)
print('Images folder {}'.format(IMAGES_FOLDER))
print('Target folder {}'.format(TARGET_FOLDER))


num_images = len(images_list)
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
  min_score_thresh = 0.5

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

  h, w, _ = image_np.shape
  y_ = []

  for idx_, score in enumerate(detections['detection_scores']):
    if score >= min_score_thresh:
      ymin, xmin, ymax, xmax = detections['detection_boxes'][idx_]
      ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
      y_.append((ymin + ymax) / 2)

  best_n_clusters = None
  cluster_score = None
  fitted = False
  try:
    for idx_c, n_clusters in enumerate(range(2, 4)):
      kmeans = KMeans(n_clusters=n_clusters)
      kmeans.fit(np.array(y_).reshape(-1, 1))
      silhouette_avg = silhouette_score(np.array(y_).reshape(-1, 1), kmeans.predict(np.array(y_).reshape(-1, 1)))
      if idx_c == 0 or cluster_score < silhouette_avg:
        best_n_clusters = n_clusters
        cluster_score = silhouette_avg
    fitted = True
  except:
    pass
  
  lines = []
  if best_n_clusters is None:
    print('Not possible to create clusters')
  else:
    kmeans = KMeans(n_clusters=best_n_clusters)
    kmeans.fit(np.array(y_).reshape(-1, 1))

    cnt_dict = {}
    counter = np.unique(kmeans.labels_, return_counts=True)
    for key, ctr in zip(counter[0], counter[1]):
      cnt_dict[key] = ctr
    
    for key in cnt_dict.keys():
      if cnt_dict[key] > 1:
        lines.append(int(kmeans.cluster_centers_.reshape(-1)[key]))
  
  img = Image.fromarray(image_np_with_detections, 'RGB')
  draw = ImageDraw.Draw(img)
  for limit in lines:
    draw.line([(0, limit), (w, limit)], width=10, fill='red')
  img.save(os.path.join(TARGET_FOLDER, 'img_{}.jpg'.format(idx)))
  # analyze only one image
  # break