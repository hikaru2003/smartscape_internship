import cv2
import numpy as np

def matching(ratio, matches, img1, kp1, img2, kp2):
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append([m])

        matches_img = cv2.drawMatchesKnn(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return good_matches

# 位置合わせを行ったグレースケール画像img2を返す
def alignment(img1, img2):
    akaze = cv2.AKAZE_create()
    
    kp1, des1 = akaze.detectAndCompute(img1, None)
    # key_img1 = cv2.drawKeypoints(img1_gray, kp1, img1)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    # key_img2 = cv2.drawKeypoints(img2_gray, kp2, img2)
    
    # 特徴のマッチング
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # 正しいマッチングのみ保持
    good_matches = matching(0.8, matches, img1, kp1, img2, kp2)
    
    # 適切なキーポイントを選択
    ref_matched_kpts = np.float32(
        [kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32(
        [kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # ホモグラフィを計算
    H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)
    
    # 画像を変換
    # img1 = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
    img2 = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1_gray, img2_gray
