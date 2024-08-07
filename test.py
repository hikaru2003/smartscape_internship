import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


dirname = os.path.dirname(__file__)
file1 = 'image3.png'

# 画像読み込む
img = cv2.imread(dirname + '/sample/' + file1)
# グレースケールに変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(dirname + '/output/gray.png', gray)

# 二値化（閾値を150に設定）
ret, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
cv2.imwrite(dirname + '/output/thr.png', binary)
# 輪郭を検出
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 全て白の画像を作成
img_blank = np.ones_like(img) * 255
# 輪郭だけを描画（黒色で描画）
img_contour_only = cv2.drawContours(img_blank, contours, -1, (0,0,0), 3)
# 描画
cv2.imwrite(dirname + '/output/result.png', img_contour_only)