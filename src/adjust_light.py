import cv2
import numpy as np
import os

# ヒストグラム平均化
def histgram_equalize(imgs, size):
    for i in range(size):
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        imgs[i] = cv2.equalizeHist(imgs[i])
        imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2RGB)
    return imgs

def adjust_brightness(image, target_mean):
    # 現在の平均明度を計算
    current_mean = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # 調整係数を計算
    if (int(target_mean) == 0 or int(current_mean) == 0):
        return image
    alpha = target_mean / current_mean
    # 明度を調整
    # print('target', target_mean)
    # print('current', current_mean)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted_image

def adjust_images(targetImage, fixImage, size):
    for i in range(size):
        target_mean = np.mean(cv2.cvtColor(targetImage[i], cv2.COLOR_BGR2GRAY))
        # 画像2の明度を画像1の平均明度に合わせて調整
        # cv2.imwrite(dirname + '/output/before00.png', split_img_2[i])
        fixImage[i] = adjust_brightness(fixImage[i], target_mean)
        # cv2.imwrite(dirname + '/output/after00.png', split_img_2[i])
    return fixImage

def adjust_image(targetImage, fixImage):
    target_mean = np.mean(cv2.cvtColor(targetImage, cv2.COLOR_BGR2GRAY))
    # 画像2の明度を画像1の平均明度に合わせて調整
    # cv2.imwrite(dirname + '/output/before00.png', split_img_2[i])
    fixImage = adjust_brightness(fixImage, target_mean)
    # cv2.imwrite(dirname + '/output/after00.png', split_img_2[i])
    return fixImage