import cv2

# imgを横にhor個, 縦にver個に分割する
def split_image(hor, ver, img):
    height, width = img.shape[:2]
    split_height = height // ver
    split_width = width // hor
    
    split_images = []
    for i in range(ver):
        for j in range(hor):
            y_start = i * split_height
            y_end = (i + 1) * split_height
            x_start = j * split_width
            x_end = (j + 1) * split_width
            split_img = img[y_start:y_end, x_start:x_end]
            split_images.append(split_img)
    
    return split_images
