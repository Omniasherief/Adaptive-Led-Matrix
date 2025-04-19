# Test 0 Phase

---

# Vehicle-Detection-LED-Matrix

This repository documents the issues encountered while exporting and running a YOLOv7-tiny model for vehicle detection (using ONNX) and mapping detection results to control an LED matrix. It contains all the code, modifications, and troubleshooting steps discussed.

## Repository Structure

```
Vehicle-Detection-LED-Matrix/
├── README.md
├── docs/
│   └── PROBLEMS_SOLUTIONS.md
├── models/
│   ├── coco.names            # Class names file (from COCO dataset)
│   ├── yolov7-tiny.pt        # Pre-trained YOLOv7-tiny weights (if needed)
│   └── yolov7-tiny.onnx      # Exported ONNX model
├── data/
│   └── test_videos/
│       └── cars_video.mp4    # Sample video for testing vehicle detection
├── src/
│   ├── export.py             # Modified export script (with safe_globals adjustments)
│   ├── utils.py              # Post-processing function (process_outputs)
│   ├── detection.py          # Functions for loading the model and running detection
│   ├── led_control.py        # Code to simulate the LED matrix (8×8 with left/right slots)
│   └── main.py               # Main script: loads video, runs detection, updates LED simulation
└── requirements.txt          # List of required Python packages
```

## README.md

# Vehicle-Detection-LED-Matrix

This project implements a real-time vehicle detection system using a YOLOv7-tiny model exported to ONNX, with detections controlling an LED matrix simulation. The project is designed to run on devices like a Raspberry Pi (with a camera and a 4-in-1 8×8 LED Matrix module) but can also be tested on a laptop using video input.

## Features
- **ONNX Export:** Export YOLOv7-tiny weights from PyTorch to ONNX while handling PyTorch 2.6 security constraints via safe_globals.
- **Vehicle Detection:** Detect vehicles (e.g., car, truck, bus, motorcycle) using a video input.
- **LED Matrix Simulation:** Simulate an 8×8 LED matrix divided into 2 halves (left/right slots) and update based on detection position.
- **Modular Code:** Organized into separate files for easy maintenance and testing.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd Vehicle-Detection-LED-Matrix
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place Model and Data:**
   - Download `coco.names` and `yolov7-tiny.pt` (if not included) into the `models/` folder.
   - Ensure the exported ONNX model (`yolov7-tiny.onnx`) is in the `models/` folder.
   - Place a sample video (e.g., `cars_video.mp4`) into `data/test_videos/`.

5. **Run the Main Script:**
   ```bash
   cd src
   python3 main.py
   ```
   Press 'q' to quit.

## Documentation

For details on problems encountered and how they were resolved, see [docs/PROBLEMS_SOLUTIONS.md](docs/PROBLEMS_SOLUTIONS.md).
```

## docs/PROBLEMS_SOLUTIONS.md

```markdown
# Problems and Solutions Documentation

This document summarizes the issues encountered during the development of the vehicle detection system and their corresponding solutions.

---

## 1. ONNX Export & Safe Globals Issues

**Problem:**  
When exporting the YOLOv7-tiny model to ONNX using PyTorch 2.6, various errors occurred due to new security defaults (i.e., `weights_only=True`). Errors like:  
- `Unsupported global: GLOBAL torch.nn.modules.container.Sequential`  
- `Unsupported global: GLOBAL models.yolo.Detect`  
- `Unsupported global: GLOBAL models.common.Conv`  
- ...and so on.

**Solution:**  
Modified the export script (`export.py`) to add the necessary safe globals. For example, at the top of `export.py`, the following code was added:
```python
import torch
from models.yolo import Model, Detect
from torch.nn.modules.container import Sequential, ModuleList
from models.common import Conv, Concat, MP, SP
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import LeakyReLU
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample

torch.serialization.add_safe_globals([
    Model, Detect, Sequential, ModuleList, Conv, Conv2d, BatchNorm2d,
    LeakyReLU, Concat, MP, SP, MaxPool2d, Upsample
])
```
Each error message was addressed by adding the corresponding class to the safe globals.

---

## 2. Output Shape & Detection Processing

**Problem:**  
The model outputs were in a grid format (e.g., shape `(1, 20, 20, 85)`) but the processing code was expecting a flat list of detections. Also, the confidence calculation was producing unexpected high or negative values.

**Solution:**  
In `utils.py`, the output grid was flattened using:
```python
detections = outputs[0].reshape(-1, outputs[0].shape[-1])
```
Then, the confidence was redefined to use only the class score (rather than multiplying by objectness), and checks were added to filter out detections with invalid confidence values.  
The function was modified to only consider detections with classes belonging to a defined set of vehicle classes.

---

## 3. Filtering Only Vehicle Detections

**Problem:**  
The model was detecting many objects, including trains, boats, and pottedplants. The goal is to detect vehicles only.

**Solution:**  
The filtering logic in `process_outputs` was updated to only keep detections whose class names are in the allowed set:
```python
vehicle_classes = {"car", "truck", "bus", "motorcycle"}
if detected_class not in vehicle_classes:
    continue
```

---

## 4. LED Matrix Simulation and Mapping

**Problem:**  
Mapping the detections to LED matrix slots was needed. The simulation should display an 8×8 grid divided into two halves (left/right), each having 4 slots. Additionally, the boxes drawn around vehicles should correspond to the correct LED slot based on their horizontal position.

**Solution:**  
The `led_control.py` file was created to simulate an 8×8 LED matrix and to "turn off" one column (slot) on the left or right based on detection position. In `main.py`, after obtaining the detection, the center_x of the bounding box is used to calculate which slot (0-3) in the left half (if center_x < width/2) or right half (if center_x ≥ width/2) should be activated.

Mapping logic example:
```python
if center_x < width/2:
    off_left_slot = int((center_x/(width/2)) * 4)
    if off_left_slot >= 4:
        off_left_slot = 3
else:
    off_right_slot = int(((center_x - (width/2))/(width/2)) * 4)
    if off_right_slot >= 4:
        off_right_slot = 3
```

---

## 5. Running on Raspberry Pi / Video Testing

**Problem:**  
The system was initially tested with a camera, but you wanted to test on a video of vehicles and eventually run it on a Raspberry Pi with a camera and a MAX7219 LED matrix module.

**Solution:**  
In `main.py`, the code was modified to read from a video file (e.g., `cars_video.mp4`) instead of a live camera stream.  
Additionally, suggestions were provided for splitting tasks between two Raspberry Pi devices if necessary (one for detection and one for LED control), with an example of using MQTT for communication.


---

## File: **requirements.txt**

```txt
opencv-python
numpy
torch
torchvision
torchaudio
pandas
requests
tqdm
pyyaml
matplotlib
onnx
onnxruntime
seaborn
scipy
onnx-graphsurgeon
paho-mqtt  # if using MQTT for inter-device communication
```

---
