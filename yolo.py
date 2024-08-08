from ultralytics import YOLO
import os
import cv2

dirname = os.path.dirname(__file__)
dirname = dirname + '/sample'

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')
    thr = 0.1
    file = input('file_type: ')
    file1 = file + '_1.png'
    file2 = file + '_2.png'
    file1_path = os.path.join(dirname, file1)
    file2_path = os.path.join(dirname, file2)
    img_1 = cv2.imread(file1_path)
    img_2 = cv2.imread(file2_path)
    
    print(f'{file1} results')
    file1_results = model.predict(file1_path, save=True, conf=0.1)
    for result in file1_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックスの座標
            confidence = int(box.conf[0])  # 信頼度
            class_id = int(box.cls[0])  # クラスID
            class_name = result.names[int(class_id)]  # クラス名
            print(f"Detected {class_name} with confidence {confidence:.2f} at location {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")

            # tmp = img_2[y1:y2, x1:x2]
            # detects = model.predict(tmp, save=True, conf=0.1, name=f"cropped_{class_name}_result")
            # print(f"detected items in {class_name}")
            # for result in detects:
            #     boxes = result.boxes
            # for box in boxes:
            #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # バウンディングボックスの座標
            #     confidence = int(box.conf[0])  # 信頼度
            #     class_id = int(box.cls[0])  # クラスID
            #     class_name = result.names[int(class_id)]  # クラス名
            #     print(f"Detected {class_name} with confidence {confidence:.2f} at location {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")

    
    print(f'{file2} results')
    file1_results = model.predict(file2_path, save=True, conf=0.1)
    for result in file1_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # バウンディングボックスの座標
            confidence = box.conf[0]  # 信頼度
            class_id = box.cls[0]  # クラスID
            class_name = result.names[int(class_id)]  # クラス名

            print(f"Detected {class_name} with confidence {confidence:.2f} at location {x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}")


    # # Predict the model
    # for filename in os.listdir(dirname):
    #     file_path = os.path.join(dirname, filename)
    #     try:
    #         model.predict(file_path, save=True, conf=0.1)
    #     except Exception as e:
    #         print(f"Error predict {file_path}: {e}")