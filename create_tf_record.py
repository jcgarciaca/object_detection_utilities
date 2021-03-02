# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Example usage:
python object_detection/dataset_tools/create_tf_record.py \
    --data_dir=/home/JulioCesar/flores/MIA2/landmark_detection \
    --output_path=/home/JulioCesar/flores/MIA2/landmark_detection/tfrecords/output_train_tf.record \
    --label_map_path=/home/JulioCesar/flores/MIA2/landmark_detection/landmarks_label_map.pbtxt \
    --output_val_path=/home/JulioCesar/flores/MIA2/landmark_detection/tfrecords/output_val_tf.record
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

from lxml import etree
import PIL.Image

import tensorflow.compat.v1 as tf
# import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

import xml.etree.ElementTree as ET
import random

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('output_val_path', '', 'Path to output val TFRecord')
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', True, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS



def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=True,
                       image_subdirectory='Images'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  
  data_dir = FLAGS.data_dir
  img_path = os.path.join(data_dir, image_subdirectory, data['filename'])
  print('img_path: ', img_path)  
  
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)  
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width, height = image.size

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:    
    if 'name' in obj:
        if obj['name'] in label_map_dict.keys():
            print('label: ', obj['name'])
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
    else:
        print('Object without name')

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      #'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      #'image/object/truncated': dataset_util.int64_list_feature(truncated),
      #'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def main(_):
  data_dir = FLAGS.data_dir
  print('data_dir:', data_dir)
  print('output_path: ', FLAGS.output_path)

  writer_train = tf.python_io.TFRecordWriter(FLAGS.output_path)
  writer_val = tf.python_io.TFRecordWriter(FLAGS.output_val_path)

  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  
  annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
  
  xml_list = os.listdir(annotations_dir + '/')
  
  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(xml_list)
  num_examples = len(xml_list)
  
  print('num_examples: ', num_examples)
  num_train = int(0.8 * num_examples) # 80 - 20 (train - eval)
  train_examples = xml_list[:num_train]
  val_examples = xml_list[num_train:]
  
  for idx, example_xml in enumerate(train_examples):
	  if idx % 100 == 0:
		  logging.info('On image %d of %d', idx, len(train_examples))
	  ### tmp because txt annotations
	  print('example_xml:', example_xml)
	  if example_xml.endswith('.txt'):
		  print('Not processed:', example_xml)
		  continue
	  path = os.path.join(annotations_dir, example_xml)
	  print('path:', path)
	  with tf.gfile.GFile(path, 'r') as fid:
		  xml_str = fid.read()
	  xml = ET.fromstring(xml_str)
	  data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
	  tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
	  writer_train.write(tf_example.SerializeToString())

  writer_train.close()
  


  for idx, example_xml in enumerate(val_examples):
	  if idx % 100 == 0:
		  logging.info('On image %d of %d', idx, len(val_examples))
	  ### tmp because txt annotations
	  print('example_xml:', example_xml)
	  if example_xml.endswith('.txt'):
		  print('Not processed:', example_xml)
		  continue
	  path = os.path.join(annotations_dir, example_xml)
	  with tf.gfile.GFile(path, 'r') as fid:
		  xml_str = fid.read()
	  xml = ET.fromstring(xml_str)
	  data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
	  tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                      FLAGS.ignore_difficult_instances)
	  writer_val.write(tf_example.SerializeToString()) 
  writer_val.close()
  


if __name__ == '__main__':
  tf.app.run()
