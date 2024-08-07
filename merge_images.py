import cv2

# 横にhor個, 縦にver個に分割されたimgを結合する
def merge_images(hor, ver, split_images):
    if len(split_images) != hor * ver:
        raise ValueError("Number of split images does not match the specified grid size")
    
    # Assuming all split images have the same size
    split_height, split_width = split_images[0].shape[:2]
    
    # Create empty canvas for the merged image
    merged_img = cv2.vconcat([cv2.hconcat(split_images[i*hor:(i+1)*hor]) for i in range(ver)])
    
    return merged_img