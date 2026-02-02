"""
Optimized Image Preprocessing Module
Fast preprocessing that won't hang on noisy images
"""

import numpy as np
from PIL import Image
import cv2
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    """Timeout exception for long-running operations"""
    pass


@contextmanager
def time_limit(seconds):
    if sys.platform.startswith("win"):
        # No SIGALRM support
        yield
    else:
        def signal_handler(signum, frame):
            raise TimeoutException("Operation timed out")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)


class ImagePreprocessor:
    """Optimized image preprocessing that won't hang"""
    
    def __init__(self):
        self.default_dpi = 300
        self.max_dimension = 4000  # Prevent processing huge images
    
    def preprocess_for_ocr(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Fast preprocessing pipeline that won't hang
        
        CRITICAL CHANGES:
        - Removed fastNlMeansDenoising (TOO SLOW)
        - Removed bilateralFilter (TOO SLOW)
        - Made deskewing optional and safer
        - Added size limits
        """
        img_array = np.array(image)
        
        # Limit image size to prevent memory issues and hangs
        img_array = self._limit_size(img_array)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Fast preprocessing pipeline
        img_array = self._fast_denoise(img_array, aggressive)
        img_array = self._enhance_contrast(img_array)
        img_array = self._sharpen(img_array, aggressive)
        img_array = self._binarize(img_array, aggressive)
        
        # Skip expensive operations for non-aggressive mode
        if aggressive:
            try:
                img_array = self._safe_deskew(img_array)
            except Exception as e:
                print(f"  ⚠ Deskewing skipped: {str(e)}")
                pass
        
        # Remove borders (fast operation)
        img_array = self._remove_borders(img_array)
        
        return Image.fromarray(img_array)
    
    def _limit_size(self, img: np.ndarray) -> np.ndarray:
        """Limit image size to prevent memory issues"""
        height, width = img.shape[:2]
        
        if height > self.max_dimension or width > self.max_dimension:
            print(f"  → Resizing large image from {width}x{height}")
            scale = self.max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"     to {new_width}x{new_height}")
        
        return img
    
    def _fast_denoise(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """
        Fast denoising using simple blur
        
        CRITICAL: Removed slow cv2.fastNlMeansDenoising and cv2.bilateralFilter
        """
        if aggressive:
            # Stronger blur for very noisy images
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            # Light blur
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Fast contrast enhancement using CLAHE"""
        # Normalize first
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply CLAHE (fast and effective)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def _sharpen(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Sharpen image to enhance text edges"""
        if aggressive:
            kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        return cv2.filter2D(img, -1, kernel)
    
    def _binarize(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Binarize using adaptive or Otsu thresholding"""
        if aggressive:
            # Otsu's method for uniform lighting
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptive threshold (better for uneven lighting)
            img = cv2.adaptiveThreshold(
                img, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=15,  # Larger block for noisy images
                C=8
            )
        return img
    
    def _safe_deskew(self, img: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        """
        Safe deskewing with limits to prevent hanging
        
        CRITICAL: Limited HoughLines to prevent hanging
        """
        try:
            # Detect edges (but don't process huge images)
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            
            # Limit HoughLines processing (CRITICAL FIX)
            lines = cv2.HoughLines(
                edges, 1, np.pi / 180, 
                threshold=100,  # Higher threshold = fewer lines
                min_theta=0, 
                max_theta=np.pi
            )
            
            if lines is None or len(lines) == 0:
                return img
            
            # CRITICAL: Only process first 20 lines (prevent hanging)
            lines = lines[:20]
            
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -max_angle < angle < max_angle:  # Only small corrections
                    angles.append(angle)
            
            if not angles:
                return img
            
            median_angle = np.median(angles)
            
            # Only rotate if angle is significant
            if abs(median_angle) > 0.5:
                (h, w) = img.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                img = cv2.warpAffine(
                    img, M, (w, h), 
                    flags=cv2.INTER_LINEAR,  # Faster than CUBIC
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        except Exception:
            # If anything fails, return original image
            pass
        
        return img
    
    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
        """Remove borders by finding largest contour"""
        try:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                margin = 10
                y_start = max(0, y - margin)
                y_end = min(img.shape[0], y + h + margin)
                x_start = max(0, x - margin)
                x_end = min(img.shape[1], x + w + margin)
                
                img = img[y_start:y_end, x_start:x_end]
        except Exception:
            pass
        
        return img
    
    def upscale_image(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
        """Upscale image to target DPI"""
        width, height = image.size
        current_dpi = image.info.get('dpi', (72, 72))
        
        if isinstance(current_dpi, tuple):
            current_dpi = current_dpi[0]
        
        scale_factor = target_dpi / current_dpi
        
        # Only upscale if needed, limit maximum
        if scale_factor > 1 and scale_factor < 5:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Don't exceed max dimension
            if new_width <= self.max_dimension and new_height <= self.max_dimension:
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def quick_preprocess(self, image: Image.Image) -> Image.Image:
        """
        Ultra-fast preprocessing for timeout-prone images
        Use this if normal preprocessing hangs
        """
        img_array = np.array(image)
        
        # Grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Limit size
        img_array = self._limit_size(img_array)
        
        # Simple contrast enhancement
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
        
        # Simple threshold
        _, img_array = cv2.threshold(
            img_array, 0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return Image.fromarray(img_array)