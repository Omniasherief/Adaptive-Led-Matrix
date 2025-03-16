# this file is just for testing yolov7-tiny.pt , yolov7-tiny.onnx after phase 0 
#!/usr/bin/env python3
import sys
import torch
import cv2

# Add YOLOv7 path to PYTHONPATH to find required files
sys.path.insert(0, "/home/omnia/yolov7")

# Import necessary functions and models
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import LoadImages

def detect(weights='/home/omnia/yolov7/yolov7-tiny.pt',
           source='/home/omnia/Adaptive Led Matrix/data/test_videos/cars_video.mp4',
           img_size=640,
           conf_thres=0.5,
           iou_thres=0.5,
           device='cpu'):

    device = torch.device(device)

    # Load the model
    model = attempt_load(weights, map_location=device)  # Removed `weights_only`
    model.eval()

    # Get class names
    names = model.module.names if hasattr(model, 'module') else model.names

    # Load source (images or video)
    dataset = LoadImages(source, img_size=img_size)
    
    for path, img, im0s, vid_cap in dataset:
        img_tensor = torch.from_numpy(img).to(device)
        img_tensor = img_tensor.float() / 255.0  # Normalize image
        img_tensor = img_tensor.unsqueeze(0)     # Add batch dimension

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    cv2.rectangle(im0s, (int(xyxy[0]), int(xyxy[1])),
                                  (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    cv2.putText(im0s, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Detection', im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()
