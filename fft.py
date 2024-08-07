import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込む
dirname = os.path.dirname(__file__)

file1 = 'diff_3_1.png'
file2 = 'diff_3_2.png'
image1 = cv2.imread(dirname + '/sample/' + file1)
image2 = cv2.imread(dirname + '/sample/' + file2)

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# フーリエ変換
f1 = np.fft.fft2(image1)
f2 = np.fft.fft2(image2)

# 周波数成分の差分
f_diff = np.abs(f1 - f2)

# 結果を表示
plt.figure(figsize=(10, 5))
plt.title('Fourier Difference')
plt.imshow(np.log(1 + f_diff), cmap='gray')
plt.show()
