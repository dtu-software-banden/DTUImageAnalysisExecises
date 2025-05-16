import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now import premade utility functions
from utils.io_utils import *  
from utils.pca_utils import * 
from utils.classifier_utils import * 
from utils.morph_utils import * 
from utils.optimization_utils import * 
from utils.plot_utils import * 


def Question1():
    print("Running Question 1")
    # Define times (in seconds)
    camera_fps = 6.25
    processing_time_per_frame = 0.230  # in seconds

    # Compute processing-limited frame rate
    processing_fps = 1 / processing_time_per_frame

    # Overall system frame rate is limited by the slower component
    system_fps = min(camera_fps, processing_fps)

    # Print result
    print(f"Camera frame rate:        {camera_fps:.2f} frames/sec")
    print(f"Processing frame rate:    {processing_fps:.2f} frames/sec")
    print(f"--> Overall system frame rate: {system_fps:.2f} frames/sec (limited by processing)")


def Question2():
    print("Running Question 2")

    # Image size and format
    width, height = 1600, 800
    channels = 3  # RGB
    bytes_per_pixel = 1  # 8 bits per channel

    # Image size in bytes
    bytes_per_image = width * height * channels * bytes_per_pixel  # = 3.84 MB
    camera_fps = 6.25

    # Total data rate from the camera over USB-2
    usb2_data_rate_bytes_per_sec = bytes_per_image * camera_fps


    # Print result
    print(f"Bytes per image: {bytes_per_image} bytes")
    print(f"Transfer rate:   {usb2_data_rate_bytes_per_sec:.2f} bytes/sec")
    print(f"USB-2 throughput used: {usb2_data_rate_bytes_per_sec/1000000:.2f} MB/s")



if __name__ == "__main__":
    Question1()
    Question2()