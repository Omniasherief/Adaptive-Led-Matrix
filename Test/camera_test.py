#import cv2

# Open the default camera (legacy mode should allow cv2.VideoCapture(0) to work)
#cap = cv2.VideoCapture(0)

#if not cap.isOpened():
 #   print("Error: Cannot open camera")
#else:
   # ret, frame = cap.read()
   # if ret:
     #   cv2.imshow("Camera Test", frame)
    #    print("Frame captured successfully. Press any key to exit.")
   #     cv2.waitKey(0)
  #  else:
 #       print("Failed to capture frame")
#cap.release()
#cv2.destroyAllWindows()

from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()

frame = picam2.capture_array()
cv2.imwrite("picamera2_test.jpg", frame)
print("Frame captured successfully using Picamera2")

picam2.stop()
