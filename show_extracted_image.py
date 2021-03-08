import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import ast

ROOT = '/home/JulioCesar/flores/MIA2/landmark_detection'
IMAGES_FOLDER = os.path.join(ROOT, 'Y_Areas_Dyn_0.03')
CSV_PATH = os.path.join(IMAGES_FOLDER, 'sections_data.csv')
TARGET_FOLDER = os.path.join(IMAGES_FOLDER, 'Alpha_Areas_003')
if not os.path.exists(TARGET_FOLDER):
  os.mkdir(TARGET_FOLDER)

df = pd.read_csv(CSV_PATH, index_col='IMG')
clusters_np = np.array(df['n_clusters'])
d_clusters = np.diff(clusters_np)
d2_clusters = np.diff(d_clusters)

tmp_array = np.insert(d2_clusters, 0, 0)
tmp_array = np.insert(tmp_array, len(tmp_array), 0)
df['diff2'] = tmp_array

non_zero_df = df[df['diff2'] != 0]

until_frame = 327
index_ref = list(non_zero_df.index)

sign = 0 # 0 - Negative. 1 - Positive
capture = False
for idx in range(54, until_frame):
  image_name = 'img_{}.jpg'.format(idx)
  print('{}/{} - {}'.format(idx + 1, until_frame, image_name))
  img = cv2.imread(os.path.join(IMAGES_FOLDER, image_name))
  blank_image = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
  if image_name in index_ref:
    if non_zero_df.loc[image_name]['diff2'] == -1:
      n_sign = 0
    elif non_zero_df.loc[image_name]['diff2'] == 1:
      n_sign = 1
    # check sign change
    if sign != n_sign:
      # change
      sign = n_sign
      capture = True if sign == 1 else False
  if capture and df.loc[image_name]['n_clusters'] == 2:
    # mask outside lines
    y1, y2 = ast.literal_eval(df.loc[image_name]['Lines'])
    draw.rectangle([(0, y1), (img.size[0], y2)], fill=255)
    img.putalpha(im_a)  
  dst = cv2.addWeighted(img, 0.7, blank_image, 0.3, 0)
  cv2.imwrite(dst, os.path.join(TARGET_FOLDER, 'img_{}.png'.format(idx)))
  # png_info = img.info
  # img.save(os.path.join(TARGET_FOLDER, 'img_{}.png'.format(idx)), **png_info)
