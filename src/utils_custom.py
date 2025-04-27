# phase 0
# utils.py
# import cv2
# import numpy as np
# import math

# def letterbox_image(img, new_shape=(640, 640), color=(114, 114, 114)):
#     """
#     Maintains the aspect ratio of the original image by adding padding
#     to reach the new_shape size without distorting the image.
#     """
#     original_h, original_w = img.shape[:2]
#     new_w, new_h = new_shape
#     scale = min(new_w / original_w, new_h / original_h)
#     resized_w = int(original_w * scale)
#     resized_h = int(original_h * scale)
    
#     print(f"[letterbox_image] Original size: {original_w}x{original_h}, "
#           f"New shape: {new_w}x{new_h}, Scale: {scale}")
#     print(f"[letterbox_image] Resized size: {resized_w}x{resized_h}")

#     resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

#     dw = new_w - resized_w
#     dh = new_h - resized_h
#     top = dh // 2
#     bottom = dh - top
#     left = dw // 2
#     right = dw - left

#     print(f"[letterbox_image] Padding (top, bottom, left, right): {top}, {bottom}, {left}, {right}")

#     letterboxed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
#     return letterboxed, scale, left, top

# def reverse_letterbox(box, scale, pad_left, pad_top):
#     """
#     Converts coordinates from the letterboxed image back 
#     to the original image before padding.
#     box = [x, y, w, h]
#     """
#     x, y, w_box, h_box = box
#     print(f"[reverse_letterbox] Original box in letterbox coords: {box} | Scale: {scale}, pad_left: {pad_left}, pad_top: {pad_top}")
#     x_new = int((x - pad_left) / scale)
#     y_new = int((y - pad_top) / scale)
#     w_new = int(w_box / scale)
#     h_new = int(h_box / scale)
#     new_box = [x_new, y_new, w_new, h_new]
#     print(f"[reverse_letterbox] Box after reverse transformation: {new_box}")
#     return new_box

# def sigmoid(x):
#     return 1 / (1 + math.exp(-x))

# def process_outputs(outputs, img_shape, conf_threshold, classes, anchor=(80, 110)):
#     boxes = []
#     confidences = []
#     class_ids = []
#     h, w = img_shape[:2]
#     detections = outputs[0].reshape(-1, outputs[0].shape[-1])
#     print("[process_outputs] Total detections (before filtering):", detections.shape[0])
    
#     allowed_vehicles = {"car", "truck", "bus", "motorcycle"}
    
#     for detection in detections:
#         scores = detection[5:]
#         if scores.size == 0:
#             continue
#         class_id = int(np.argmax(scores))
#         if class_id >= len(classes):
#             continue
        
#         detected_class = classes[class_id].lower()
#         if detected_class not in allowed_vehicles:
#             continue
        
#         confidence = scores[class_id]
#         if confidence <= 0 or confidence > 1:
#             continue
        
#         if confidence > conf_threshold:
#             # Apply activation functions to center coordinates
#             cx = sigmoid(detection[0]) * w
#             cy = sigmoid(detection[1]) * h
#             # Apply exp to width and height and use anchor values
#             anchor_w, anchor_h = anchor
#             w_box = math.exp(detection[2]) * anchor_w
#             h_box = math.exp(detection[3]) * anchor_h
            
#             x = int(cx - w_box / 2)
#             y = int(cy - h_box / 2)
#             boxes.append([x, y, int(w_box), int(h_box)])
#             confidences.append(float(confidence))
#             class_ids.append(class_id)
#             print(f"[process_outputs] Detected: {detected_class} | Confidence: {confidence}")
#             print(f"[process_outputs] Box (letterbox coords): Center ({cx:.2f}, {cy:.2f}), "
#                   f"Width: {w_box:.2f}, Height: {h_box:.2f} -> Top-left: ({x}, {y})")
    
#     print("[process_outputs] Total detections (after filtering):", len(boxes))
#     return boxes, confidences, class_ids

#-------------------------------------------------------------------------------------------------------------------------------
#phase 1
#-------------------------------------------------------------------------------------------------------------------------------
# #!/usr/bin/env python3
# import sys
# import os
# import torch
# import cv2
# import numpy as np
# from utils.general import non_max_suppression, scale_coords
# from models.yolo import Model, Detect
# from torch.nn.modules.container import Sequential, ModuleList
# from models.common import Conv, MP, Concat, SP
# from torch.nn.modules.conv import Conv2d
# from torch.nn.modules.batchnorm import BatchNorm2d
# from torch.nn.modules.activation import LeakyReLU
# from torch.nn.modules.pooling import MaxPool2d
# from torch.nn.modules.upsampling import Upsample

# torch.serialization.add_safe_globals([
#     Model, Detect, Sequential, ModuleList, Conv, MP, Concat, SP,
#     Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Upsample
# ])

# import sys
# import os
# import torch
# import cv2
# import numpy as np

# # Add YOLOv7 repository path to PYTHONPATH so that required modules can be found
# sys.path.insert(0, os.path.abspath("/home/omnia/yolov7"))

# def load_model(weights, device):
#     """
#     Loads the YOLOv7-tiny model using torch.hub.load from the GitHub repo.
#     Moves the model to the specified device (CPU/GPU) and sets it to evaluation mode.
    
#     For best practices, we use torch.hub.load which internally manages safe_globals.
#     """
#     model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights, trust_repo=True)
#     model.to(device).eval()
#     return model

# def preprocess_image(img, device):
#     """
#     Converts an image (numpy array) to a Tensor suitable for model input:
#       - Converts to tensor.
#       - Normalizes pixel values (0-1).
#       - Adds a batch dimension.
#     """
#     img_tensor = torch.from_numpy(img).to(device)
#     img_tensor = img_tensor.float() / 255.0  # Normalize the image
#     img_tensor = img_tensor.unsqueeze(0)     # Add batch dimension
#     return img_tensor




#-------------------------------------------------------------------------------------------------------------------------------
#phase 2
#-----------------------------------------------------------------------------------------------------------------------------==
#change pc user to rasp user omnia-->omnia2
#!/usr/bin/env python3
import sys
import os
import torch
import cv2
import numpy as np
from utils.general import non_max_suppression, scale_coords
from models.yolo import Model, Detect
from torch.nn.modules.container import Sequential, ModuleList
from models.common import Conv, MP, Concat, SP
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample

torch.serialization.add_safe_globals([
    Model, Detect, Sequential, ModuleList, Conv, MP, Concat, SP,
    Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, Upsample
])



# Add YOLOv7 repository path to PYTHONPATH so that required modules can be found
sys.path.insert(0, os.path.abspath("/home/omnia2/yolov7"))

def load_model(weights, device):
    """
    Loads the YOLOv7-tiny model using torch.hub.load from the GitHub repo.
    Moves the model to the specified device (CPU/GPU) and sets it to evaluation mode.
    
    For best practices, we use torch.hub.load which internally manages safe_globals.
    """
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', weights, trust_repo=True)
    model.to(device).eval()
    return model

def preprocess_image(img, device):
    """
    Converts an image (numpy array) to a Tensor suitable for model input:
      - Converts to tensor.
      - Normalizes pixel values (0-1).
      - Adds a batch dimension.
    """
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalize the image
    img_tensor = img_tensor.unsqueeze(0)     # Add batch dimension
    return img_tensor
