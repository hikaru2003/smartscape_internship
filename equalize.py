import cv2
import os
import numpy as np
from IPython import display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)

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
min_height = height / 30
min_width = width / 30
print(height)
print(width)

img_1 = cv2.resize(img_1, (int(width), int(height)))

img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
# cv2.imwrite(dirname + '/output/gray_3_1.png', img_1_gray)
# cv2.imwrite(dirname + '/output/gray_3_2.png', img_2_gray)

# 位置合わせ処理
akaze = cv2.AKAZE_create()
kp1, des1 = akaze.detectAndCompute(img_1_gray, None)
# key_img_1 = cv2.drawKeypoints(img_1_gray, kp1, img_1)
kp2, des2 = akaze.detectAndCompute(img_2_gray, None)
# key_img_2 = cv2.drawKeypoints(img_2_gray, kp2, img_2)
# cv2.imwrite(dirname + '/output/keypoints_img_1.jpg', key_img_1)
# cv2.imwrite(dirname + '/output/keypoints_img_2.jpg', key_img_2)

# 特徴のマッチング
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 正しいマッチングのみ保持
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

# matches_img = cv2.drawMatchesKnn(
#     img_1,
#     kp1,
#     img_2,
#     kp2,
#     good_matches,
#     None,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imwrite(dirname + '/output/matches.jpg', matches_img)

# 適切なキーポイントを選択
ref_matched_kpts = np.float32(
    [kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
sensed_matched_kpts = np.float32(
    [kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# ホモグラフィを計算
H, status = cv2.findHomography(
    ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

# 画像を変換
warped_image_1 = cv2.warpPerspective(img_1, H, (img_1.shape[1], img_1.shape[0]))
warped_image_2 = cv2.warpPerspective(img_2, H, (img_2.shape[1], img_2.shape[0]))

# cv2.imwrite(dirname + '/output/warped_1.jpg', warped_image_1)
# cv2.imwrite(dirname + '/output/warped_2.jpg', warped_image_2)

img_g = cv2.cvtColor(warped_image_1, cv2.COLOR_BGR2GRAY)
img_2_gray = cv2.cvtColor(warped_image_2, cv2.COLOR_BGR2GRAY)

# cv2.imwrite(dirname + '/output/gray_1.jpg', img_1_gray)
# cv2.imwrite(dirname + '/output/gray_2.jpg', img_2_gray)

# with cv2
eq_g = cv2.equalizeHist(img_g)

# visualize
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Equalized image with cv2
ax1.imshow(eq_g, cmap="gray")
ax1.set_title("Equalized image with cv2")
ax1.axis('off')

ax2.imshow(img_g, cmap="gray")
ax2.set_title("Non Equalized image")
ax2.axis('off')

# Histogram
ax3.hist(eq_g.ravel(), 256, [0,256], label="cv2")
ax3.set_title("Density histogram")
ax3.legend()

plt.tight_layout()
plt.show()
