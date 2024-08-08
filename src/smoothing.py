import cv2

# スムージング
def smoothing(img1, img2, height, width):
    hsize = height // 50
    wsize = width // 50
    # hsize = 5
    # wsize = 5
    hsize = hsize if hsize % 2 == 1 else hsize + 1
    wsize = wsize if wsize % 2 == 1 else wsize + 1
    # if (hsize > 51): hsize = 51
    # if (wsize > 51): wsize = 51
    # hsize = 5
    # wsize = 5
    print('hsize', hsize)
    print('wsize', wsize)
    return cv2.GaussianBlur(img1,(wsize, hsize),0),\
           cv2.GaussianBlur(img2,(wsize, hsize),0)