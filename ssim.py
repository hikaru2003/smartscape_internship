import cv2
import os
import numpy as np
from IPython import display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

dirname = os.path.dirname(__file__)

# 画像読み込み
# file1 = input('file1: ')
# file2 = input('file2: ')
file1 = 'diff_3_1.png'
file2 = 'diff_3_2.png'
image1 = cv2.imread(dirname + '/sample/' + file1)
image2 = cv2.imread(dirname + '/sample/' + file2)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 画像サイズを取得
height1, width1 = image1.shape
height2, width2 = image2.shape

# 小さい方のサイズを選択
new_height = min(height1, height2)
new_width = min(width1, width2)

# 両方の画像をリサイズ
image1 = cv2.resize(image1, (new_width, new_height))
image2 = cv2.resize(image2, (new_width, new_height))


# SSIMを計算
score, diff = ssim(image1, image2, full=True)
diff = (diff * 255).astype("uint8")

# 差分画像を二値化
_, diff_binary = cv2.threshold(diff, 127, 255, cv2.THRESH_BINARY_INV)

# 結果を表示
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image 1')
plt.imshow(image1, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Original Image 2')
plt.imshow(image2, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Difference Image (SSIM)')
plt.imshow(diff_binary, cmap='gray')
plt.show()