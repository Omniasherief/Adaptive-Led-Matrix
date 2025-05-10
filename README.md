
---

# Adaptive Headlights 💡

Night driving presents unique challenges: restricted visibility, sudden glare from oncoming vehicles, and the constant need to balance illumination with safety. Although nighttime travel accounts for only 25% of all driving time, it tragically contributes to 50% of traffic fatalities.¹ In Bhopal, India alone, high-beam glare caused 1,470 accidents in 2024.² According to the National Safety Council, the risk of traffic deaths at night is three times greater than during daylight hours.³

Inspired by Audi’s cutting-edge adaptive headlight technology, Adaptive LED Matrix offers a proof-of-concept solution that is both smart and cost-effective. By analyzing real-time road conditions via a camera, the system dynamically adjusts an LED display to optimize beam patterns—minimizing glare for other drivers while maximizing visibility for the user. This repository is organized into two phases:

- **Phase 1: Software Simulation**  
  Simulate adaptive LED behavior using YOLOv7 for real-time object detection and OpenCV for LED matrix visualization.

- **Phase 2: Hardware Integration (Work in Progress)**  
  Integrate the simulation with actual hardware components, such as MAX7219 LED modules and a Raspberry Pi Camera Board Version 2 (Sony IMX219 sensor).

---
## Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#System-Architecture)
- [Features and Options](#features-and-options)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Sample Video Outputs](#sample-video-outputs)
- [Challenges and Solutions](#challenges-and-solutions)
- [Camera and Hardware Details](#camera-and-hardware-details)
- [Hardware Setup](#hardware-setup)
- [Output](#output)
- [External Links and Resources](#external-links-and-resources)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Project Overview

The Adaptive LED Matrix project simulates a dynamic lighting system that responds to vehicles detected in a video stream. By using YOLOv7, the system maps the detected vehicle positions to specific LED slots on an 8×8 matrix. In Phase 1, the entire process is simulated in software—laying the foundation for eventual hardware integration (Phase 2).
<p align="center">
  <img src="docs/Adaptive_led_matrix.png" alt="Adaptive Headlights">
</p>
 
<p align="center">
  <img src="docs/with_Adaptive.png" alt="docs/with_Adaptive.png">
</p>

---
## System Architecture

<p align="center">
  <img src="docs/Adaptive_led_matrix.png" alt="Adaptive LED Matrix">
</p>
 
<p align="center">
  <img src="docs/with_Adaptive.png" alt="docs/with_Adaptive.png">
</p>

---


## Features and Options

### Detection Options
- **Input Source:**  
  The system accepts either a video file or a live camera feed.  
  - **Video:** For example, use `data/test_videos/cars_video.mp4`.  
  - **Live Camera:** Set `source = 0` (or integrate with libraries like `picamera2`).

- **Image Size (`img_size`):**  
  Controls the resolution provided to YOLOv7. Lower resolutions (e.g., 320) offer faster inference at a potential cost in detection accuracy.

- **Thresholds:**  
  - **Confidence (`conf_thres`):** Minimum confidence level for a detection to be considered.  
  - **IoU (`iou_thres`):** Intersection over Union threshold used in Non-Max Suppression to remove overlapping detections.

### LED Matrix Simulation Options
- **Matrix Layout:**  
  The 8×8 LED matrix is simulated as two modules:
  - **Left Module:** Columns 0–3.
  - **Right Module:** Columns 4–7.

- **Dynamic LED Control:**  
  For each frame, the system determines which LED slots should be turned off based on the position of detected vehicles. If multiple vehicles are detected on the same side, all corresponding slots are updated in real time.

### YOLOv7 and ONNX
- **YOLOv7:**  
  Chosen for its high accuracy and real-time detection performance.
- **ONNX (Future Option):**  
  Although the current implementation uses PyTorch, converting the model to ONNX is planned to enable faster inference and broader device compatibility.

### Camera Resolution Options
The Raspberry Pi Camera Board Version 2 supports multiple resolutions:
- **1080p at 30 FPS (Full HD)**
- **720p at 60 FPS (HD)**
- **640×480 at 90 FPS (VGA)**

These options help balance image quality and processing speed.

---

## Repository Structure
```
Adaptive-LED-Matrix/
├── data/
│   ├── test_images/         # Images for debugging and testing object detection
│   └── test_videos/         # Sample videos used for simulation (e.g., cars_video.mp4, realv.mp4, highway.mp4)
├── docs/
│   ├── 00000.png               # which is used in repo
│   └── problems_solutions.md  # Documentation of encountered issues and their solutions , 0 phase which is commented 
├── models/                  # YOLOv7 model files and ONNX export scripts
├── screen/
│   ├── yolo7.png            #yolo7 is downloaded
│   ├── shiftToraspb.png     # from pc to raspb
|   └── downloadvid.png      # out put of script/Download_YoutubeVid.py
├── src/
│   ├── main.py              # Main script for running the simulation
│   ├── detection.py         # Object detection and video processing using YOLOv7
│   ├── led_control.py       # LED matrix simulation/control logic
│   └── utils_custom.py      # Utility functions (model loading, image preprocessing)
├── scripts/
|   └──Download_YoutubeVid.py
├── Test/
│   ├── camera_test.py       # Testing camera integration and object detection
│   ├── test_ledmatrix.py    # Testing LED matrix control logic
│   └── spidev-test.py       # Testing SPI communication for hardware control
├── requirements.txt         # Python dependencies (e.g., torch, opencv-python, numpy)
├── README.md                # Project documentation
└── LICENSE    
```
### data Link

Access data folder via the following Google Drive folder:

[Data](https://drive.google.com/drive/folders/1CEXMNom0yxXcvw-b2lsECXVmUIV8u60k?usp=drive_link)

---

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/omnia/Adaptive-LED-Matrix.git
   cd Adaptive-LED-Matrix
   ```

2. **Install Dependencies for Phase 1:**

   Navigate to the Phase 1 folder:

   ```bash
   cd src
   pip install -r requirements.txt
   ```

   _Note: Ensure you are using a compatible Python version (e.g., Python 3.8 – 3.11). If you experience version conflicts (e.g., NumPy requirements), adjust the dependency versions in `requirements.txt` accordingly._

3. **(Phase 2)**  
   Additional dependencies for hardware control will be included in `phase2/requirements.txt` as development progresses.

---

## Usage

To run the simulation for Phase 1, execute the main script:

```bash
python3 src/main.py #--weights yolov7-tiny.pt --source "data/test_videos/cars_video.mp4" --img-size 640 --conf-thres 0.5 --iou-thres 0.5 --device cpu --view-img

```
---

## Sample Video Outputs

The system processes multiple video sources and updates the LED matrix in real time. For instance:

1. **Urban Roads:**  
   The system detects vehicles and dynamically disables LED slots to simulate adaptive headlight patterns in a city setting.  
   [🎥 Watch urban.mp4](docs/urban.mp4)

2. **Highway Traffic:**  
   With a higher FPS input, the system smoothly updates the LED matrix even at high speeds.  
   [🎥 Watch Highway.mp4](docs/Highway.mp4)

3. **Controlled Test Scenario:**  
   A controlled video feed used to test different resolutions and LED mapping strategies.  
   [🎥 Watch Countryside.mp4](docs/Countryside.mp4)

Video outputs are available in the `docs/` folder.

---

## Challenges and Solutions

### Version Conflicts
- **Problem:**  
  Incompatibilities between certain versions of NumPy, Python, and ONNX.
- **Solution:**  
  Specify compatible versions (e.g., `numpy>=1.18.5,<1.24.0`) in `requirements.txt` and consult the YOLOv7 repository for recommendations.

### Model Export and Integration
- **Problem:**  
  Converting YOLOv7 models to ONNX for deployment can be challenging.
- **Solution:**  
  Use YOLOv7’s export scripts (e.g., `export.py`) and test with ONNX Runtime to ensure proper conversion.

### Hardware Component Choices
- **MAX7219 Module 4-in-1 8X8 LED Matrix Module:**  
  Selected for its simplicity and cost-effectiveness in controlling an 8×8 LED array.
- **Raspberry Pi Camera Board V2:**  
  Provides versatile resolution options to balance between quality and performance.

---

## Camera and Hardware Details
- **Raspberry Pi 4**
  [Product Link](https://www.amazon.eg/-/en/Xute-Raspberry-4GB-Cortex-A72-Bluetooth/dp/B0C7KXMP7W/ref=asc_df_B0C7KXMP7W/?tag=egoshpadde-21&linkCode=df0&hvadid=725192583796&hvpos=&hvnetw=g&hvrand=8937843028174914309&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9218366&hvtargid=pla-2223870482804&mcid=ee88c1e71fac3b13ba66ee8b066c4386&language=en_AE&gad_source=1&th=1)

- **Raspberry Pi Camera Board Version 2**  
 [Product Link](https://makerselectronics.com/product/raspberry-pi-camera-board-version-2-sony-imx219-8-megapixel-sensor?srsltid=AfmBOops-2ER_4kj-9x-s9uZMQzArf3JvOZPqDYmQLbiWg1jf0ydw9vr)  
  - Supports 1080p @ 30 FPS, 720p @ 60 FPS, and 640×480 @ 90 FPS.
  
- **MAX7219 LED Matrix Module**  
  [Product Link](https://makerselectronics.com/product/max7219-module-4-in-1-8x8-led-matrix-module-green?srsltid=AfmBOoqv8gfw7miqCEhIlVbLhiq07I1p3okgKziEIgnZbUZe4yzHcQt2)  
  - Ideal for controlling 8×8 LED arrays, these modules provide a straightforward way to simulate dynamic lighting.

---
## Hardware setup
### Wiring the 4-in-1 MAX7219 LED Matrix to the Raspberry Pi

Connect each MAX7219 module on the hat to the Pi’s SPI pins as follows:

| Module   | MAX7219 Pin | Raspberry Pi Pin   | GPIO #               | Notes                              |
| -------- | ----------- | ----------------- | -------------------- | ---------------------------------- |
| **Left** | VCC         | Pin 1 (3.3 V)     | —                    | Or 5 V if required by your module  |
|          | GND         | Pin 6 (GND)       | —                    |                                    |
|          | DIN         | Pin 19            | GPIO 10 (MOSI)       |                                    |
|          | CLK         | Pin 23            | GPIO 11 (SCLK)       |                                    |
|          | CS          | Pin 24            | GPIO 8 (CE0)         |                                    |
| **Right**| VCC         | Pin 1 (3.3 V)     | —                    | Or 5 V                              |
|          | GND         | Pin 6 (GND)       | —                    |                                    |
|          | DIN         | Pin 19            | GPIO 10 (MOSI)       | Shared data line                  |
|          | CLK         | Pin 23            | GPIO 11 (SCLK)       | Shared clock                       |
|          | CS          | Pin 26            | GPIO 7 (CE1)         |                                    |

---

### Verifying the SPI Interface

1. **Enable SPI**  
   ```bash
   sudo raspi-config
   # → Interface Options → SPI → Yes
   ```

2. **Check kernel modules**  
   ```bash
   lsmod | grep spi
   ```
   You should see:
   ```
   spidev                 16384  4
   spi_bcm2835            20480  0
   ```

3. **List SPI devices**  
   ```bash
   ls /dev/spidev*
   ```
   Expected:
   ```
   /dev/spidev0.0  /dev/spidev0.1
   ```

---

### Installing Required Libraries

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-pip libfreetype6-dev libjpeg-dev
pip3 install luma.led_matrix
```

---

### Testing SPI with `spidev-test`

1. **Clone & build**  
   ```bash
   git clone https://github.com/rm-hull/spidev-test.git
   cd spidev-test
   make
   ```

2. **Run**  
   ```bash
   sudo ./spidev_test -D /dev/spidev0.0
   ```
   Expected:
   ```
   spi mode: 0x0
   bits per word: 8
   max speed: 500000 Hz (500 KHz)
   RX | 00 00 … 00 | …
   ```

---

### Running the Python Test

```bash
python3 Test/test_ledmatrix.py
```

This will light up your matrices to verify the `luma.led_matrix` driver.

---

### Debugging Tips

- **Wiring**: Double-check all pin mappings.  
- **SPI Enabled**: Confirm via `raspi-config`.  
- **spidev-test**: Use C test to isolate hardware issues.  
- **Power**: Ensure a stable 3.3 V/5 V supply per module spec.  
- **Virtual Env**: Isolate Python deps if needed.

---

### Additional Resources

- [Luma.LED_Matrix Docs](https://luma-led-matrix.readthedocs.io/en/latest/)  
- [MAX7219 4-in-1 Tutorial](https://www.instructables.com/User-Manual-MAX7219-Dot-Matrix-4-in-1/)  
- [Controlling MAX7219 on Pi (YouTube)](https://www.youtube.com/watch?v=6M7L8U36cO0)  

---

## Output

### 1. Output
![Output](docs/me.png)

### 2. “Close Slots” Output
![Close Slots Output](docs/close_slots.png)

---

## External Links and Resources

- [YOLOv7 GitHub Repository](https://github.com/WongKinYiu/yolov7)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Letterboxing in YOLO (Medium Article)](https://medium.com/@reachraktim/letterboxing-in-yolov5-yolov7-yolov8-an-intuitive-explanation-with-python-code-88f7d4323d6c)
- [which yolo](https://youtu.be/QOC6vgnWnYo?si=vThQYaN5HXmn4OA8)  
- [Additional YouTube Videos](https://youtu.be/SfqN-Hc5two?si=CdvGx0RcCg9JzO_g), [Video 3](https://youtu.be/WgPbbWmnXJ8?si=PHV8jY3v1FccKsgx), [What is YOLO algorithm](https://youtu.be/ag3DLKsl2vk?si=Ad01mu_K90sdmX1I), [best model / best algorithm](https://youtu.be/ce5PBam-V4Y?si=iwPSFvcWG9q_0Eak)
- [Reddit Discussion on Object Detection Models](https://www.reddit.com/r/computervision/comments/1gm40zi/how_yolo_or_other_object_detection_model_handle/?chainedPosts=t3_1hvnxld)


---

## Future Enhancements

### Phase 2 – Hardware Integration
- **Real Hardware:**  
  Integrate MAX7219 LED modules with a Raspberry Pi for physical LED control.
- **Camera Integration:**  
  Use live camera feeds and experiment with different resolutions.
- **ONNX Conversion:**  
  Convert the YOLOv7 model to ONNX for optimized inference on embedded devices.
- **Enhanced Multi-Object Handling:**  
  Refine LED mapping logic to handle overlapping detections more robustly.
- **User Interface:**  
  Develop a GUI for easier system configuration and monitoring.

### Software Enhancements
- **Advanced Logging and Metrics:**  
  Implement detailed logging to monitor system performance and detection accuracy.
- **Energy Consumption and Intelligent Brightness Control:**
  Adopt more energy-efficient technologies to ensure that the system can operate for extended periods without requiring frequent maintenance. Additionally,          instead of turning off a portion of the LED matrix when an object is detected, implement an intelligent brightness control mechanism that adjusts the LED          intensity based on the distance of the approaching vehicle.
- **Modular Improvements:**  
  Continue refactoring code for improved maintainability and scalability.

---

## License

This project is licensed under the MIT License.

---
## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

---
## Contact

📧 omniasheriefahmed@gmail.com


