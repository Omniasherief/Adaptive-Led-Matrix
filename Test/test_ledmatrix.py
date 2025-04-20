#!/usr/bin/env python3
"""
Debug-capable test code for two 4-in-1 MAX7219 LED matrix modules.
This script includes detailed debug output and checks to diagnose SPI, device dimensions,
and rendering operations step by step.
"""
import sys
import time
from luma.core.interface.serial import spi, noop
from luma.led_matrix.device import max7219
from luma.core.render import canvas

# Helper for debug logging
def dbg(msg):
    print(f"[DEBUG] {msg}")

# Initialize SPI and device with debug
def init_device(port, device_id, cascaded=4, orientation=0, rotate=0):
    dbg(f"Initializing SPI on port={port}, device={device_id}")
    try:
        serial = spi(port=port, device=device_id, gpio=noop())
        device = max7219(serial, cascaded=cascaded,
                         block_orientation=orientation, rotate=rotate)
        dbg(f"-> Device initialized: cascaded={device.cascaded}, "
            f"width={device.width}, height={device.height}")
        # Set medium contrast for visibility
        device.contrast(64)
        dbg("-> Contrast set to 64")
        return device
    except Exception as e:
        dbg(f"Failed to initialize device {device_id}: {e}")
        sys.exit(1)

# Test drawing operations on a single pixel to verify basic functionality
def test_single_pixel(device, side):
    dbg(f"{side} - Testing single pixel toggle")
    try:
        with canvas(device) as draw:
            draw.point((0, 0), fill="white")
        dbg(f"{side} - White pixel drawn at (0,0)")
        time.sleep(1)
        with canvas(device) as draw:
            draw.point((0, 0), fill="black")
        dbg(f"{side} - Pixel cleared at (0,0)")
        time.sleep(1)
    except Exception as e:
        dbg(f"{side} - Pixel test failed: {e}")
        sys.exit(1)

# Test cycling through each 8x8 slot
def test_slots(device, side, delay=2):
    slot_width = 8

    dbg(f"{side} - Full display ON")
    with canvas(device) as draw:
        draw.rectangle((0, 0, device.width-1, device.height-1), fill="white")
    time.sleep(delay)

    for slot in range(device.cascaded):
        dbg(f"{side} - Turning off slot {slot}")
        with canvas(device) as draw:
            draw.rectangle((0, 0, device.width-1, device.height-1), fill="white")
            x_start = slot * slot_width
            # debug rectangle coordinates
            dbg(f"{side} - Drawing black rect at x={x_start} to x={x_start + slot_width-1}")
            draw.rectangle((x_start, 0, x_start + slot_width-1, device.height-1), fill="black")
        time.sleep(delay)

    dbg(f"{side} - Resetting full display ON")
    with canvas(device) as draw:
        draw.rectangle((0, 0, device.width-1, device.height-1), fill="white")
    time.sleep(delay)
    dbg(f"{side} - Slot test complete.")


def main():
    dbg("Script start")
    left = init_device(port=0, device_id=0)
    right = init_device(port=0, device_id=1)

    # Verify device dimensions
    dbg(f"Left dimensions: {left.width}×{left.height}")
    dbg(f"Right dimensions: {right.width}×{right.height}")

    test_single_pixel(left, "Left")
    test_slots(left, "Left")

    test_single_pixel(right, "Right")
    test_slots(right, "Right")

    dbg("All tests complete. Exiting.")

if __name__ == "__main__":
    # Suggest running with sudo and unbuffered output for best debug visibility
    dbg("Run as: sudo python3 -u <script_name>.py")
    main()
