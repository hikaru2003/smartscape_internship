import cv2
import os
import numpy as np
from IPython import display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

# Delete all files in dirname/output
output_dir = dirname  + '/output'
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
        
# 画像読み込み
# file1 = input('file1: ')
# file2 = input('file2: ')
file1 = 'diff_3_1.png'
file2 = 'diff_3_2.png'
img_1 = cv2.imread(dirname + '/sample/' + file1)
img_2 = cv2.imread(dirname + '/sample/' + file2)


# 画像の処理
height = img_2.shape[0]
width = img_2.shape[1]
print('height', height)
print('width', width)
min_height = height / 30
min_width = width / 30

img_1 = cv2.resize(img_1, (int(width), int(height)))

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
cv2.imwrite(dirname + '/output/gray1.png', img_1_gray)
cv2.imwrite(dirname + '/output/gray2.png', img_2_gray)

# ヒストグラム平均化
# img_1_gray = cv2.equalizeHist(img_1_gray)
# img_2_gray = cv2.equalizeHist(img_2_gray)

# スムージング
hsize = height // 50
wsize = width // 50
hsize = hsize if hsize % 2 == 1 else hsize + 1
wsize = wsize if wsize % 2 == 1 else wsize + 1
img_1_gray = cv2.GaussianBlur(img_1_gray,(wsize, hsize),0)
img_2_gray = cv2.GaussianBlur(img_2_gray,(wsize, hsize),0)
cv2.imwrite(dirname + '/output/blur_1.jpg', img_1_gray)
cv2.imwrite(dirname + '/output/blur_2.jpg', img_2_gray)

# ２画像の差異を計算
# img_diff = warped_image_1.astype(int) - warped_image_2.astype(int)

# 画像を引き算
img_diff = cv2.absdiff(img_1_gray, img_2_gray)
cv2.imwrite(dirname + '/output/img_diff.png', img_diff)

# 2値化
ret2, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite(dirname + '/output/img_th.png', img_th)

# 輪郭を検出
contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
image1_contours = img_1.copy()
cv2.drawContours(image1_contours, contours, -1, (0, 0, 255), 2)

# 画像を生成
cv2.imwrite(dirname + '/output/diff.png', image1_contours)