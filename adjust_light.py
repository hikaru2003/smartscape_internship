import cv2
import numpy as np

# ヒストグラム平均化
def histgram_equalize(img):
    return cv2.equalizeHist(img)

def adjust_brightness(image, target_mean):
    # 現在の平均明度を計算
    current_mean = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # 調整係数を計算
    alpha = target_mean / current_mean
    # beta = target_mean % current_mean
    # 明度を調整
    # print('target', target_mean)
    # print('current', current_mean)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def adjust_image(targetImage, fixImage, size):
    for i in range(size):
        target_mean = np.mean(cv2.cvtColor(targetImage[i], cv2.COLOR_BGR2GRAY))
        # 画像2の明度を画像1の平均明度に合わせて調整
        # cv2.imwrite(dirname + '/output/before00.png', split_img_2[i])
        fixImage[i] = adjust_brightness(fixImage[i], target_mean)
        # cv2.imwrite(dirname + '/output/after00.png', split_img_2[i])
    return fixImage