import cv2
import numpy as np

def create_diff_img(dirname, img1, img1_gray, img2_gray):
    # 画像サイズの調整
    img1_gray = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))
    img1 = cv2.resize(img1, (img2_gray.shape[1], img2_gray.shape[0]))
    # 画像を引き算
    img_diff = cv2.absdiff(img1_gray, img2_gray)
    cv2.imwrite(dirname + '/output/img_diff.png', img_diff)
    # 2値化
    ret2, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)
    img_th_rgb = cv2.cvtColor(img_th, cv2.COLOR_GRAY2RGB)
    img_th_rgb = np.where(img_th_rgb == (255, 255, 255), (0, 0, 255), img1)
    cv2.imwrite(dirname + '/output/img_th.png', img_th)
    cv2.imwrite(dirname + '/output/diff_filled.png', img_th_rgb)
    # 輪郭を検出
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0, 0, 255), 2)
    # 画像を生成
    cv2.imwrite(dirname + '/output/diff.png', img1)