"""
Image Preprocessing Module
Advanced image preprocessing for OCR accuracy improvement
"""

import numpy as np
from PIL import Image
import cv2


class ImagePreprocessor:
    """Advanced image preprocessing for OCR accuracy improvement"""
    
    def __init__(self):
        self.default_dpi = 300
    
    def preprocess_for_ocr(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """Comprehensive preprocessing pipeline for OCR"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        img_array = self._denoise(img_array, aggressive)
        img_array = self._enhance_contrast(img_array)
        img_array = self._sharpen(img_array, aggressive)
        img_array = self._binarize(img_array, aggressive)
        img_array = self._deskew(img_array)
        img_array = self._remove_borders(img_array)
        
        return Image.fromarray(img_array)
    
    def _denoise(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        
        img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        
        return img
    
    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    
    def _sharpen(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        return cv2.filter2D(img, -1, kernel)
    
    def _binarize(self, img: np.ndarray, aggressive: bool = False) -> np.ndarray:
        if aggressive:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        return img
    
    def _deskew(self, img: np.ndarray) -> np.ndarray:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None and len(lines) > 0:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 0.5:
                    (h, w) = img.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
        return img
    
    def _remove_borders(self, img: np.ndarray) -> np.ndarray:
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
        
        return img
    
    def upscale_image(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
        width, height = image.size
        current_dpi = image.info.get('dpi', (72, 72))[0]
        scale_factor = target_dpi / current_dpi
        
        if scale_factor > 1:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image