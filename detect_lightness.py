import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from src.adjust_light import histgram_equalize, adjust_brightness, adjust_image, adjust_images
from src.split_image import split_image
from src.merge_images import merge_images
from src.smoothing import smoothing
from src.homography import homography

dirname = os.path.dirname(__file__)

def main():
    # 画像読み込み
    while True:
        file = input('file_type: ')
        if file.lower() == 'eof':
            break
        file1 = file + '_1.png'
        file2 = file + '_2.png'
        img_1 = cv2.imread(dirname + '/sample/' + file1)
        img_2 = cv2.imread(dirname + '/sample/' + file2)
        img1_mean = np.mean(cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY))
        img2_mean = np.mean(cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY))
        print(img1_mean, img2_mean)

if __name__ == "__main__":
    main()
