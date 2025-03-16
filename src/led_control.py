#phase 0 
# led_control.py
# import cv2
# import numpy as np

# def simulate_led_slots(off_left_slot=None, off_right_slot=None, window_name="LED Matrix Slots Simulation"):
#     """
#     Simulate an 8x8 LED matrix divided into two halves:
#       - Left half (columns 0-3) represent left slots.
#       - Right half (columns 4-7) represent right slots.
#     The specified slot will be "turned off" (black).
    
#     Parameters:
#       off_left_slot: Slot number (0-3) on the left side to turn off.
#       off_right_slot: Slot number (0-3) on the right side to turn off.
#     """
#     cell_size = 30
#     rows, cols = 8, 8
#     sim_img = np.full((rows * cell_size, cols * cell_size, 3), 255, dtype=np.uint8)
    
#     if off_left_slot is not None and 0 <= off_left_slot < 4:
#         col = off_left_slot
#         x_start = col * cell_size
#         sim_img[:, x_start:x_start+cell_size] = 0
    
#     if off_right_slot is not None and 0 <= off_right_slot < 4:
#         col = 4 + off_right_slot
#         x_start = col * cell_size
#         sim_img[:, x_start:x_start+cell_size] = 0
    
#     # Draw grid lines
#     for i in range(rows+1):
#         y = i * cell_size
#         cv2.line(sim_img, (0, y), (cols * cell_size, y), (0, 0, 0), 1)
#     for j in range(cols+1):
#         x = j * cell_size
#         cv2.line(sim_img, (x, 0), (x, rows * cell_size), (0, 0, 0), 1)
    
#     cv2.imshow(window_name, sim_img)


#------------------------------------------------------------------------------------------------------

#phase 1
#!/usr/bin/env python3
import cv2
import numpy as np

def simulate_led_slots(active_left=None, active_right=None, window_name="LED Matrix Simulation"):
    """
    Simulates two 8x8 LED matrices (one for the left side and one for the right side)
    using OpenCV. Columns 0-3 represent the left matrix; columns 4-7 represent the right matrix.
    
    Parameters:
      active_left: Integer or list of integers (0-3) indicating which column(s) in the left matrix are turned off.
      active_right: Integer or list of integers (0-3) indicating which column(s) in the right matrix are turned off.
      
    NOTE:
      For real hardware using MAX7219 modules, replace the simulation below with hardware-specific calls.
    """
    cell_size = 30
    rows, cols = 8, 8
    sim_img = np.full((rows * cell_size, cols * cell_size, 3), 255, dtype=np.uint8)
    
    # Process left LED matrix (columns 0-3)
    if active_left is not None:
        if isinstance(active_left, (list, tuple, set)):
            for slot in active_left:
                if 0 <= slot < 4:
                    x_start = slot * cell_size
                    sim_img[:, x_start:x_start+cell_size] = 0
        else:
            if 0 <= active_left < 4:
                x_start = active_left * cell_size
                sim_img[:, x_start:x_start+cell_size] = 0
    
    # Process right LED matrix (columns 4-7)
    if active_right is not None:
        if isinstance(active_right, (list, tuple, set)):
            for slot in active_right:
                if 0 <= slot < 4:
                    x_start = (4 + slot) * cell_size
                    sim_img[:, x_start:x_start+cell_size] = 0
        else:
            if 0 <= active_right < 4:
                x_start = (4 + active_right) * cell_size
                sim_img[:, x_start:x_start+cell_size] = 0

    # Draw grid lines for clarity
    for i in range(rows+1):
        y = i * cell_size
        cv2.line(sim_img, (0, y), (cols * cell_size, y), (0, 0, 0), 1)
    for j in range(cols+1):
        x = j * cell_size
        cv2.line(sim_img, (x, 0), (x, rows * cell_size), (0, 0, 0), 1)
    
    cv2.imshow(window_name, sim_img)
    cv2.waitKey(1)
    
