#phase 0 
# import cv2
# from utils import letterbox_image, reverse_letterbox, process_outputs

# def load_yolo_model(model_path):
#     """
#     Loads the YOLO ONNX model.
#     """
#     net = cv2.dnn.readNetFromONNX(model_path)
#     return net

# def detect_objects(net, frame, input_size=(640, 640), conf_threshold=0.9, nms_threshold=0.1, classes=None):
#     """
#     Runs object detection on the given frame:
#       1. Applies letterbox transformation.
#       2. Converts the image to blob.
#       3. Runs the model.
#       4. Processes the outputs.
#       5. Reverses the letterbox transformation.
#       6. Applies Non-Maximum Suppression (NMS).
#     """
#     # Letterbox transformation
#     letterbox_img, scale, pad_left, pad_top = letterbox_image(frame, new_shape=input_size)
    
#     # Prepare blob from letterboxed image
#     blob = cv2.dnn.blobFromImage(letterbox_img, 1/255.0, input_size, swapRB=True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward()
    
#     # Process outputs in letterbox coordinates
#     boxes, confidences, class_ids = process_outputs(outputs, letterbox_img.shape, conf_threshold, classes)
    
#     # Reverse letterbox transformation (pass original image shape as (h, w))
#     for i in range(len(boxes)):
#         boxes[i] = reverse_letterbox(boxes[i], scale, pad_left, pad_top)
    
#     # Apply Non-Maximum Suppression (NMS)
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
#     return boxes, confidences, class_ids, indices
#-----------------------------------------------------------------------------
#phase 1

#!/usr/bin/env python3
import torch
import cv2
from utils_custom import preprocess_image
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords

def detect_objects(model, source, img_size, conf_thres, iou_thres, device):
    """
    Processes frames (from a video file or live camera) using YOLOv7-tiny.
    
    Steps:
      - Loads images/frames using LoadImages.
      - Preprocesses each frame.
      - Runs inference with the model.
      - Applies Non-Maximum Suppression.
      - Scales bounding box coordinates to match the original frame.
    
    Returns a list of tuples:
      (bounding_box, confidence, class, original_frame, x_center)
    where x_center is the horizontal center of the bounding box.
    """
    dataset = LoadImages(source, img_size=img_size)
    detections = []
    
    for path, img, im0s, vid_cap in dataset:
        img_tensor = preprocess_image(img, device)
        
        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    x_center = (x1 + x2) / 2  # Compute horizontal center
                    detections.append((xyxy, conf, cls, im0s, x_center))
    
    return detections
