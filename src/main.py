#phase 0
# import cv2
# import os
# from detection import load_yolo_model, detect_objects
# from led_control import simulate_led_slots
# from utils import letterbox_image  # only if you want to debug letterboxing

# # Load class names from coco.names (assumed to be in ../models/)
# classes_path = os.path.join(os.path.dirname(__file__), "../models/coco.names")
# with open(classes_path, "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# # Set model path (assumed to be in ../models/)
# model_path = os.path.join(os.path.dirname(__file__), "../models/yolov7-tiny.onnx")
# net = load_yolo_model(model_path)

# # Choose input source:
# # For video file:
# video_path = os.path.join(os.path.dirname(__file__), "../data/test_videos/way.mp4")
# cap = cv2.VideoCapture(video_path)

# # For real-time camera on Raspberry Pi, comment out the video lines and use:
# # cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video/camera.")
#     exit()

# print("Starting detection. Press 'q' to quit.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run detection on the frame using letterbox processing inside detect_objects.
#     boxes, confidences, class_ids, indices = detect_objects(net, frame, input_size=(640,640), conf_threshold=0.91, nms_threshold=0.1, classes=classes)
    
#     # Calculate LED slot mapping based on detection center.
#     off_left_slot = None
#     off_right_slot = None
#     height, width, _ = frame.shape

#     if len(indices) > 0:
#         # For simplicity, we take the first detection (or you can implement a selection strategy)
#         for i in indices.flatten():
#             x, y, w_box, h_box = boxes[i]
#             center_x = x + w_box // 2
#             cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
#             cv2.putText(frame, classes[class_ids[i]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
#             if center_x < width / 2:
#                 off_left_slot = int((center_x / (width/2)) * 4)
#                 if off_left_slot >= 4:
#                     off_left_slot = 3
#             else:
#                 off_right_slot = int(((center_x - (width/2)) / (width/2)) * 4)
#                 if off_right_slot >= 4:
#                     off_right_slot = 3
#             # You may break after first valid detection if desired.
#             break

#     cv2.imshow("Detection", frame)
#     simulate_led_slots(off_left_slot=off_left_slot, off_right_slot=off_right_slot, window_name="LED Matrix Slots Simulation")
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------------------------------------------------
# phase 1
#-----------------------------------------------------------------------------------------------------------------------------
# #!/usr/bin/env python3
# import sys
# import os
# import torch
# import cv2
# from utils_custom import load_model, preprocess_image
# from led_control import simulate_led_slots

# # Add YOLOv7 path to PYTHONPATH so that 'utils' and other modules are found
# sys.path.insert(0, os.path.abspath("/home/omnia/yolov7"))

# # Detection settings
# weights = '/home/omnia/yolov7/yolov7-tiny.pt'
# source = '/home/omnia/Adaptive Led Matrix/data/test_videos/realvv.mp4'
# img_size = 640
# conf_thres = 0.65
# iou_thres = 0.6
# device = torch.device('cpu')

# # Load the model (using attempt_load method from utils_custom)
# model = load_model(weights, device)
# names = model.module.names if hasattr(model, 'module') else model.names

# # Import YOLOv7 utilities for detection
# from utils.datasets import LoadImages
# from utils.general import non_max_suppression, scale_coords

# # Define a maximum width for display (if the frame is large, it will be resized)
# MAX_WIDTH = 1280

# # Load video frames using LoadImages
# dataset = LoadImages(source, img_size=img_size)

# for path, img, im0s, vid_cap in dataset:
#     # Optionally resize frame if it's too large for display
#     if im0s.shape[1] > MAX_WIDTH:
#         scale = MAX_WIDTH / im0s.shape[1]
#         new_width = int(im0s.shape[1] * scale)
#         new_height = int(im0s.shape[0] * scale)
#         im0s = cv2.resize(im0s, (new_width, new_height))
    
#     # Preprocess the frame for detection
#     img_tensor = torch.from_numpy(img).to(device)
#     img_tensor = img_tensor.float() / 255.0  # Normalize image
#     img_tensor = img_tensor.unsqueeze(0)     # Add batch dimension

#     # Run model inference
#     with torch.no_grad():
#         pred = model(img_tensor)[0]
#         pred = non_max_suppression(pred, conf_thres, iou_thres)

#     # Initialize sets for LED slots for current frame (each side: left/right, 4 slots each)
#     left_slots = set()
#     right_slots = set()
#     frame_width = im0s.shape[1]

#     for det in pred:
#         if len(det):
#             # Adjust box coordinates to original frame size
#             det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0s.shape).round()
#             for *xyxy, conf, cls in det:
#                 x1, y1, x2, y2 = [int(x) for x in xyxy]
#                 x_center = (x1 + x2) / 2.0

#                 # Determine LED slot within the corresponding half (each half is divided into 4 slots)
#                 if x_center < frame_width / 2:
#                     # Left half: compute ratio and slot (0-3)
#                     ratio = x_center / (frame_width / 2)
#                     slot = int(ratio * 4)
#                     if slot > 3:
#                         slot = 3
#                     left_slots.add(slot)
#                 else:
#                     # Right half: compute ratio and slot (0-3)
#                     ratio = (x_center - frame_width / 2) / (frame_width / 2)
#                     slot = int(ratio * 4)
#                     if slot > 3:
#                         slot = 3
#                     right_slots.add(slot)

#                 # Draw bounding box and label on the frame
#                 label = f'{names[int(cls)]} {conf:.2f}'
#                 cv2.rectangle(im0s, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(im0s, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Convert slot sets to lists (or use None if no detection on a side)
#     active_left = list(left_slots) if left_slots else None
#     active_right = list(right_slots) if right_slots else None

#     # Update LED matrix simulation based on current frame's detections
#     simulate_led_slots(active_left, active_right)
    
#     cv2.imshow("Detection", im0s)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()


#-----------------------------------------------------------------------------------------------------------------------------
# phase 2
#-----------------------------------------------------------------------------------------------------------------------------

#!/usr/bin/env python3
import sys, os
import torch
import cv2
from picamera2 import Picamera2

# Make sure YOLOv7 repo is importable
sys.path.insert(0, os.path.abspath("/home/omnia2/yolov7"))

from detection import detect_objects
from utils_custom   import load_model
from led_control    import update_led_modules

# Model & thresholds tuned for tiny
WEIGHTS    = '/home/omnia2/yolov7/yolov7-tiny.pt'
IMG_SIZE   = 640
CONF_THRES = 0.35
IOU_THRES  = 0.50
DEVICE     = torch.device('cpu')

# Load model
model = load_model(WEIGHTS, DEVICE)
names = model.module.names if hasattr(model, 'module') else model.names

# Configure Picamera2 for 3-channel BGR output
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"format": "BGR888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

frame_count = 0
try:
    while True:
        frame = picam2.capture_array()  # shape (480,640,3)
        if frame is None:
            continue

        frame_count += 1
        im0s = frame.copy()
        h, w = im0s.shape[:2]

        # Run detection on this single frame
        detections = detect_objects(
            model, frame,
            img_size   = IMG_SIZE,
            conf_thres = CONF_THRES,
            iou_thres  = IOU_THRES,
            device     = DEVICE
        )

        # Map detections to LED slots
        left_slots, right_slots = set(), set()
        for x1, y1, x2, y2, conf, cls, x_center in detections:
            if x_center < w/2:
                slot = min(int((x_center/(w/2))*4), 3)
                left_slots.add(slot)
            else:
                slot = min(int(((x_center-w/2)/(w/2))*4), 3)
                right_slots.add(slot)

            # Debug draw
            cv2.rectangle(im0s, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(im0s, f"{names[cls]} {conf:.2f}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        print(f"[Frame {frame_count}] left={left_slots}, right={right_slots}")

        # Update LED matrices
        update_led_modules(
            left_active_slots  = list(left_slots)  or None,
            right_active_slots = list(right_slots) or None
        )

        # Show debug window
        cv2.imshow("Detection", im0s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
