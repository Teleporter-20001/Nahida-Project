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

    # Helper to get frame size from first available image
    def get_frame_size(files):
        for f in files:
            img = cv2.imread(f)
            if img is not None:
                return img.shape
        raise FileNotFoundError("No valid image files found to determine frame size.")

    # Create videos for frame_*.png and debug_*.png
    timestamp = int(time.time())
    for files, out_name in [(frame_files, f'output_video_{timestamp}.mp4'), (debug_files, f'debug_video_{timestamp}.mp4')]:
        if files:
            height, width, layers = get_frame_size(files)
            # determine fourcc in a compatible way for different OpenCV builds
            if hasattr(cv2, 'VideoWriter_fourcc'):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:
                # fallback for older OpenCV bindings (may not exist in all installs)
                try:
                    fourcc = cv2.cv.FOURCC(*'mp4v')
                except Exception:
                    fourcc = 0
            fps = 20
            out = cv2.VideoWriter(out_name, fourcc, fps, (width, height))
            for frame_file in files:
                img = cv2.imread(frame_file)
                if img is None:
                    # skip unreadable frames
                    continue
                # ensure frame size matches the video writer
                if (img.shape[1], img.shape[0]) != (width, height):
                    img = cv2.resize(img, (width, height))
                out.write(img)
            out.release()
            print(f"Video saved as {out_name} with {len(files)} frames at {fps} FPS.")

    # remove these pictures to trash
    for file in frame_files:
        os.remove(file)
    for file in debug_files:
        os.remove(file)