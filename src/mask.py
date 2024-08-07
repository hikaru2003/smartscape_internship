import cv2

# マスクを作成する関数
def create_mask(dirname, warped_image):
    # 黒い部分を検出
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cv2.imwrite(dirname + '/output/mask.png', mask)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)