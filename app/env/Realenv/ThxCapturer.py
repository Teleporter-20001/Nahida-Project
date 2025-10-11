from typing import Optional, Tuple
import dxcam
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import threading

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Frame resizing quality may be reduced.")

try:
    import win32gui
    import win32process
    import win32con
    import win32api
    import psutil
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("Warning: win32api and psutil not available. Window detection will be limited.")

def get_dpi_scale_factor() -> float:
    """Get the current DPI scale factor.
    
    Returns:
        float: DPI scale factor (1.0 for 100%, 1.25 for 125%, 1.5 for 150%, etc.)
    """
    if not WIN32_AVAILABLE:
        return 1.0
    
    try:
        # Try to set process DPI aware first
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
        
        # Get the DPI of the primary monitor
        hdc = win32gui.GetDC(0)
        try:
            # Try different methods to get DPI
            import ctypes
            dpi_x = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX = 88
        except:
            # Fallback method
            dpi_x = 96  # Default DPI
        finally:
            win32gui.ReleaseDC(0, hdc)
        
        # Standard DPI is 96, so scale factor is dpi/96
        scale_factor = dpi_x / 96.0
        return scale_factor
    except Exception as e:
        print(f"Warning: Could not get DPI scale factor: {e}")
        return 1.0

def find_process_window(process_name: str) -> Optional[Tuple[int, int, int, int]]:
    """Find the window of a specific process and return its client area coordinates.
    
    Args:
        process_name (str): The name of the process (e.g., "th11c.exe")
        
    Returns:
        Optional[Tuple[int, int, int, int]]: Client area bounds (left, top, right, bottom) or None if not found
    """
    if not WIN32_AVAILABLE:
        return None
        
    def enum_windows_callback(hwnd, windows):
        if win32gui.IsWindowVisible(hwnd):
            # Get window title
            window_title = win32gui.GetWindowText(hwnd)
            
            # Get process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            try:
                # Get process name
                process = psutil.Process(pid)
                if process.name().lower() == process_name.lower():
                    # Get client rectangle (content area without borders/title bar)
                    try:
                        client_rect = win32gui.GetClientRect(hwnd)
                        # Convert client coordinates to screen coordinates
                        client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))
                        client_right = client_left + client_rect[2]
                        client_bottom = client_top + client_rect[3]
                        
                        windows.append({
                            'hwnd': hwnd,
                            'title': window_title,
                            'rect': (client_left, client_top, client_right, client_bottom),
                            'pid': pid,
                            'client_size': (client_rect[2], client_rect[3])
                        })
                    except Exception as e:
                        # Fallback to window rectangle if client rect fails
                        window_rect = win32gui.GetWindowRect(hwnd)
                        windows.append({
                            'hwnd': hwnd,
                            'title': window_title,
                            'rect': window_rect,
                            'pid': pid,
                            'client_size': None
                        })
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    
    if windows:
        # Return the first matching window's client rectangle
        best_window = windows[0]
        print(f"Found window '{best_window['title']}' with client area: {best_window['rect']}")
        if best_window['client_size']:
            print(f"Client size: {best_window['client_size']}")
        return best_window['rect']
    
    return None

def run(process_name: str, shm_name: str, width: int, height: int, capture_freq: int = 60):
    """Run the capture process to capture frames from the process window and store them in shared memory.

    Args:
        process_name (str): The name of the process to capture.
        shm_name (str): The name of the shared memory block.
        width (int): The width of the capture area.
        height (int): The height of the capture area.
        capture_freq (int, optional): The frequency of capturing frames. Defaults to 60.
    """
    shm = None
    camera = None
    lock_shm = None
    lock_name = f"{shm_name}_lock"
    
    try:
        # Connect to existing shared memory block (created by parent process)
        shm = shared_memory.SharedMemory(name=shm_name)
        
        # Create a lock for synchronizing access to shared memory
        # Using a simple flag approach since multiprocessing.Lock can't be shared by name
        try:
            lock_shm = shared_memory.SharedMemory(name=lock_name, create=True, size=1)
        except FileExistsError:
            lock_shm = shared_memory.SharedMemory(name=lock_name)
        
        # Initialize DXcam camera
        camera = dxcam.create()
        
        # Try to find the target window
        target_region = None
        last_window_check = 0
        window_check_interval = 5.0  # Check window position every 5 seconds
        dpi_scale = get_dpi_scale_factor()
        current_capture_width = max(1, int(round(width * dpi_scale)))
        current_capture_height = max(1, int(round(height * dpi_scale)))
        
        # Calculate frame interval
        frame_interval = 1.0 / capture_freq
        
        print(f"Starting capture for process: {process_name}")
        print(f"DPI scale factor: {dpi_scale}")
        
        while True:
            start_time = time.time()
            
            # Periodically check for window position
            if start_time - last_window_check > window_check_interval:
                dpi_scale = get_dpi_scale_factor()
                window_rect = find_process_window(process_name)
                scaled_width = max(1, int(round(width * dpi_scale)))
                scaled_height = max(1, int(round(height * dpi_scale)))
                if window_rect:
                    left, top, right, bottom = window_rect
                    
                    # The coordinates from find_process_window are already client area coordinates
                    # We need to determine the actual capture region based on our desired size
                    client_width = max(1, right - left)
                    client_height = max(1, bottom - top)
                    
                    print(f"Window client area: {client_width}x{client_height} at ({left}, {top})")
                    
                    # Use the full client area dimensions for capture (scaled back later if needed)
                    capture_width = client_width
                    capture_height = client_height
                    
                    if capture_width > 0 and capture_height > 0:
                        # Center the capture area within the client area if needed
                        offset_x = max(0, (client_width - capture_width) // 2)
                        offset_y = max(0, (client_height - capture_height) // 2)
                        
                        target_region = (
                            left + offset_x,
                            top + offset_y,
                            left + offset_x + capture_width,
                            top + offset_y + capture_height
                        )
                        current_capture_width = capture_width
                        current_capture_height = capture_height
                        print(f"Capture region: {target_region} (size: {capture_width}x{capture_height})")
                    else:
                        target_region = None
                        current_capture_width = scaled_width
                        current_capture_height = scaled_height
                        print(f"Window too small for capture: {client_width}x{client_height}")
                else:
                    target_region = None
                    current_capture_width = scaled_width
                    current_capture_height = scaled_height
                    print(f"Window not found for process: {process_name}, using full screen")
                
                last_window_check = start_time
            
            # Capture frame
            try:
                if target_region:
                    frame = camera.grab(region=target_region)
                else:
                    # Fallback to full screen if window not found
                    frame = camera.grab()
                    if frame is not None:
                        h, w = frame.shape[:2]
                        crop_h = min(current_capture_height, h)
                        crop_w = min(current_capture_width, w)
                        start_y = max(0, (h - crop_h) // 2)
                        start_x = max(0, (w - crop_w) // 2)
                        frame = frame[start_y:start_y + crop_h, start_x:start_x + crop_w]
            except Exception as e:
                print(f"Frame capture error: {e}")
                frame = None
            
            if frame is not None:
                try:
                    # Ensure frame has correct dimensions
                    if frame.shape[:2] != (height, width):
                        if CV2_AVAILABLE:
                            interpolation = cv2.INTER_AREA if frame.shape[0] > height or frame.shape[1] > width else cv2.INTER_LINEAR
                            frame = cv2.resize(frame, (width, height), interpolation=interpolation)
                        else:
                            # Fallback to numpy resize (less accurate) if OpenCV unavailable
                            frame = np.resize(frame, (height, width, frame.shape[2] if frame.ndim == 3 else 1))
                    
                    # Convert to uint8 if necessary
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Ensure RGB format (3 channels)
                    if len(frame.shape) == 3 and frame.shape[2] >= 3:
                        frame_data = frame[:, :, :3].flatten()
                    else:
                        print(f"Invalid frame shape: {frame.shape}")
                        continue
                    
                    # Write frame data to shared memory with simple locking
                    if len(frame_data) * frame_data.itemsize <= shm.size:
                        # Simple spinlock using shared memory flag
                        max_wait_time = 0.001  # 1ms max wait
                        wait_start = time.time()
                        
                        while lock_shm.buf[0] == 1:  # Wait if locked
                            if time.time() - wait_start > max_wait_time:
                                break
                            time.sleep(0.0001)  # 0.1ms sleep
                        
                        # Acquire lock
                        lock_shm.buf[0] = 1
                        
                        try:
                            # Write data
                            frame_bytes = frame_data.astype(np.uint8).tobytes()
                            shm.buf[:len(frame_bytes)] = frame_bytes
                        finally:
                            # Release lock
                            lock_shm.buf[0] = 0
                            
                except Exception as e:
                    print(f"Frame processing error: {e}")
            
            # Control capture frequency
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("Capture process interrupted by user")
    except Exception as e:
        print(f"Capture process error: {e}")
    finally:
        # Cleanup
        if lock_shm is not None:
            try:
                lock_shm.close()
            except:
                pass
        if shm is not None:
            try:
                shm.close()
            except:
                pass
        if camera is not None:
            try:
                camera.release()
            except:
                pass
        print("Capture process cleanup completed")



class ThxCapturer:
    
    _instance: Optional['ThxCapturer'] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ThxCapturer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        self.image = None
        self.capture_target: str = "th11c.exe"
        self.target_window_width: int = 640
        self.target_window_height: int = 480
        self.capture_process: Optional[mp.Process] = None
        self.shm: Optional[shared_memory.SharedMemory] = None
        self.lock_shm: Optional[shared_memory.SharedMemory] = None
        self._lock: threading.Lock = threading.Lock()
        self._shm_name: str = "thx_frame_shm"
        self._lock_name: str = f"{self._shm_name}_lock"
        
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from shared memory.
        
        Returns:
            numpy.ndarray or None: The latest frame as a numpy array (height, width, 3) or None if no frame available.
        """
        if self.shm is None or self.lock_shm is None:
            return None
            
        try:
            with self._lock:
                # Simple spinlock to wait for writer to finish
                max_wait_time = 0.01  # 10ms max wait
                wait_start = time.time()
                
                while self.lock_shm.buf[0] == 1:  # Wait if locked by writer
                    if time.time() - wait_start > max_wait_time:
                        break
                    time.sleep(0.0001)  # 0.1ms sleep
                
                # Read frame data from shared memory
                frame_bytes = bytes(self.shm.buf)
                
                # Convert bytes back to numpy array
                frame_data = np.frombuffer(frame_bytes, dtype=np.uint8)
                
                # Reshape to original image dimensions
                expected_size = self.target_window_width * self.target_window_height * 3
                if len(frame_data) >= expected_size:
                    frame = frame_data[:expected_size].reshape(
                        (self.target_window_height, self.target_window_width, 3)
                    )
                    return frame.copy()  # Return a copy to avoid shared memory issues
                else:
                    return None
                    
        except Exception as e:
            print(f"Error reading frame from shared memory: {e}")
            return None
    
    def start(self):
        """Start the frame capture process."""
        if self.capture_process is not None and self.capture_process.is_alive():
            print("Capture process is already running.")
            return
            
        try:
            # Create shared memory block for frame data
            shm_size = self.target_window_width * self.target_window_height * 3
            self.shm = shared_memory.SharedMemory(
                name=self._shm_name, 
                create=True, 
                size=shm_size
            )
            
            # Create shared memory block for locking (1 byte flag)
            self.lock_shm = shared_memory.SharedMemory(
                name=self._lock_name,
                create=True,
                size=1
            )
            # Initialize lock to 0 (unlocked)
            self.lock_shm.buf[0] = 0
            
            # Start capture process
            self.capture_process = mp.Process(
                target=run, 
                args=(
                    self.capture_target, 
                    self.shm.name, 
                    self.target_window_width, 
                    self.target_window_height,
                    60  # capture frequency
                )
            )
            self.capture_process.start()
            print(f"Started capture process for {self.capture_target}")
            
        except Exception as e:
            print(f"Error starting capture process: {e}")
            self._cleanup()
            raise
        
    def stop(self):
        """Stop the frame capture process and clean up resources."""
        print("Stopping capture process...")
        
        # Stop the capture process
        if self.capture_process is not None:
            try:
                self.capture_process.terminate()
                self.capture_process.join(timeout=5.0)  # Wait up to 5 seconds
                if self.capture_process.is_alive():
                    print("Warning: Capture process did not terminate gracefully")
                else:
                    print("Capture process terminated successfully")
            except Exception as e:
                print(f"Error stopping capture process: {e}")
            finally:
                self.capture_process = None
        
        # Clean up shared memory
        self._cleanup()
        
    def _cleanup(self):
        """Clean up shared memory resources."""
        if self.shm is not None:
            try:
                self.shm.close()
                self.shm.unlink()
                print("Frame shared memory cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up frame shared memory: {e}")
            finally:
                self.shm = None
        
        if self.lock_shm is not None:
            try:
                self.lock_shm.close()
                self.lock_shm.unlink()
                print("Lock shared memory cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up lock shared memory: {e}")
            finally:
                self.lock_shm = None
    
    def is_running(self) -> bool:
        """Check if the capture process is currently running.
        
        Returns:
            bool: True if the capture process is running, False otherwise.
        """
        return (self.capture_process is not None and 
                self.capture_process.is_alive() and 
                self.shm is not None and
                self.lock_shm is not None)
    
    def set_target_process(self, process_name: str):
        """Set the target process name to capture.
        
        Args:
            process_name (str): The name of the process (e.g., "th11c.exe")
            
        Note:
            This method can only be called when the capture process is not running.
        """
        if self.is_running():
            raise RuntimeError("Cannot change target process while capture is running. Stop first.")
        self.capture_target = process_name
        
    def set_capture_size(self, width: int, height: int):
        """Set the capture area size.
        
        Args:
            width (int): Width of the capture area
            height (int): Height of the capture area
            
        Note:
            This method can only be called when the capture process is not running.
        """
        if self.is_running():
            raise RuntimeError("Cannot change capture size while capture is running. Stop first.")
        self.target_window_width = width
        self.target_window_height = height
        
    def get_capture_info(self) -> dict:
        """Get current capture configuration information.
        
        Returns:
            dict: Configuration information including target process, size, and status
        """
        return {
            'target_process': self.capture_target,
            'capture_width': self.target_window_width,
            'capture_height': self.target_window_height,
            'is_running': self.is_running(),
            'shared_memory_name': self._shm_name,
            'win32_available': WIN32_AVAILABLE
        }
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.stop()
        except:
            pass  # Ignore errors during cleanup in destructor