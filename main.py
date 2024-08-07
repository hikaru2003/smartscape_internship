import cv2
import os
import numpy as np
from IPython import display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from adjust_light import histgram_equalize, adjust_brightness, adjust_image
from split_image import split_image
from merge_images import merge_images
from smoothing import smoothing
from homography import get_sift_matches, find_homography_opencv,\
    apply_homography, blend_image, homography

dirname = os.path.dirname(__file__)

def diff_img(img1, img1_gray, img2_gray):
    # 画像サイズの調整
    img1_gray = cv2.resize(img1_gray, (img2_gray.shape[1], img2_gray.shape[0]))
    # 画像を引き算
    img_diff = cv2.absdiff(img1_gray, img2_gray)
    cv2.imwrite(dirname + '/output/img_diff.png', img_diff)
    # 2値化
    ret2, img_th = cv2.threshold(img_diff, 40, 255, cv2.THRESH_BINARY)
    cv2.imwrite(dirname + '/output/img_th.png', img_th)
    # 輪郭を検出
    contours, hierarchy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0, 0, 255), 2)
    # 画像を生成
    cv2.imwrite(dirname + '/output/diff.png', img1)

# マスクを作成する関数
def create_mask(warped_image):
    # 黒い部分を検出
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(dirname + '/output/mask.png', mask)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

def main():
    # Parameters ---------------
    seed = 0
    n_matches = 100
    max_iters = 5000
    verbose = 2
    # --------------------------
    # /output 内のファイルを削除
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
    file1 = 'image3.png'
    file2 = 'image4.png'
    img_1 = cv2.imread(dirname + '/sample/' + file1)
    img_2 = cv2.imread(dirname + '/sample/' + file2)
    IMG_1 = img_1
    IMG_2 = img_2

    # 画像サイズを同じにする
    h1, w1, c1 = img_1.shape
    h2, w2, c2 = img_2.shape
    height = h1 if h1 < h2 else h2
    width = w1 if w1 < w2 else w2
    print('height', height)
    print('width', width)
    min_height = height / 30
    min_width = width / 30
    img_1 = cv2.resize(img_1, (int(width), int(height)))
    img_2 = cv2.resize(img_2, (int(width), int(height)))
    
    # グレースケール変換
    # img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    # img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    
    # ホモグラフィ変換
    # img_2_gray = homography(img_2, img_1, n_matches, max_iters)
    img_2 = homography(img_2, img_1, n_matches, max_iters)
    cv2.imwrite(dirname + '/output/homo_2.png', img_2)
    mask = create_mask(img_2)
    # img_1 = cv2.resize(img_1, (mask.shape[1], mask.shape[0]))
    img_1 = cv2.bitwise_and(img_1, mask)
    
    # imgを横にhor個, 縦にver個に分割する
    hor = 4
    ver = 3
    split_img_1 = split_image(hor, ver, img_1)
    split_img_2 = split_image(hor, ver, img_2)

    # ヒストグラム平均化 二つの画像で大きく異なるので、逆に精度が落ちる
    # img_1_gray = histgram_equalize(img_1_gray)
    # img_2_gray = histgram_equalize(img_2_gray)
    # cv2.imwrite(dirname + '/output/hist_1.png', img_1_gray)
    # cv2.imwrite(dirname + '/output/hist_2.png', img_2_gray)
    
    # 平均明度調整
    # 明度が高いほうの画像を低いほうの画像に合わせる <- そのほうが明るさの違いによる誤差分が小さくなる
    img1_mean = np.mean(cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY))
    img2_mean = np.mean(cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY))
    if (img1_mean < img2_mean):
        split_img_2 = adjust_image(split_img_1, split_img_2, ver*hor)
    else:
        split_img_1 = adjust_image(split_img_2, split_img_1, ver*hor)

    # 画像の結合
    img_1 = merge_images(hor, ver, split_img_1)
    img_2 = merge_images(hor, ver, split_img_2)
    
    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    
    # smoothing
    img_1_gray, img_2_gray = smoothing(img_1_gray, img_2_gray, height, width)
    cv2.imwrite(dirname + '/output/blur_1.jpg', img_1_gray)
    cv2.imwrite(dirname + '/output/blur_2.jpg', img_2_gray)

    # ２画像の差異を計算
    # img_1 = cv2.resize(img_1, (img_2.shape[1], img_2.shape[0]))
    # img_diff = img_1.astype(int) - img_2.astype(int)
    # cv2.imwrite(dirname + '/output/color_diff.png', img_diff)
    # 差分画像の生成
    diff_img(IMG_1, img_1_gray, img_2_gray)

if __name__ == "__main__":
    main()
