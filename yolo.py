from ultralytics import YOLO
import os

dirname = os.path.dirname(__file__)
dirname = dirname + '/sample'

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')

    # Predict the model
    for filename in os.listdir(dirname):
        file_path = os.path.join(dirname, filename)
        try:
            model.predict(file_path, save=True, conf=0.1)
        except Exception as e:
            print(f"Error predict {file_path}: {e}")
