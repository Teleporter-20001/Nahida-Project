import time
import cv2
import glob
import os

if __name__ == '__main__':

    # Get all frame_*.png and debug_*.png files sorted by filename
    frame_files = sorted(glob.glob('./frame_*.png'))
    debug_files = sorted(glob.glob('./debug_*.png'))

    if not frame_files and not debug_files:
        raise FileNotFoundError("No frame_*.png or debug_*.png files found in the current directory.")

    # remove these pictures to trash
    for file in frame_files:
        os.remove(file)
    for file in debug_files:
        os.remove(file)