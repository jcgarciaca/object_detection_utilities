import cv2
import numpy as np
import os

ROOT = '/home/JulioCesar/flores/MIA2/landmark_detection'
IMAGES_FOLDER = os.path.join(ROOT, 'Y_Areas')
num_frames = len(os.listdir(IMAGES_FOLDER))
dst_video = os.path.join(ROOT, 'videos', 'video.mp4')

frame_array = []
fps = 30

for num in range(num_frames):
	print('Processing frame {} / {}'.format(num + 1, num_frames))
	filename = os.path.join(IMAGES_FOLDER, 'img_{}.jpg'.format(num))
	img = cv2.imread(filename)
	img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
	height, width, _ = img.shape
	size = (width, height)
	frame_array.append(img)

out = cv2.VideoWriter(dst_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
for frame in frame_array:
	out.write(frame)
out.release()
