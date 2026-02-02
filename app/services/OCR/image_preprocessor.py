"""
Ultimate ImagePreprocessor - Final Version
Better quality detection + raw image fallback
"""

import numpy as np
from PIL import Image
import cv2


class ImagePreprocessor:
    """
    Production-ready preprocessor with:
    - More aggressive quality detection
    - Multiple preprocessing attempts
    - Raw image fallback if preprocessing makes it worse
    """
    
    def __init__(self):
        self.default_dpi = 300
        self.max_dimension = 4000
    
    def preprocess_for_ocr(self, image: Image.Image, aggressive: bool = False) -> Image.Image:
        """
        Main preprocessing with automatic quality detection
        Returns the BEST version for OCR (preprocessed OR raw)
        """
        try:
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            # Store original for comparison
            original_gray = img_gray.copy()
            
            # Limit size
            img_gray = self._limit_size(img_gray)
            
            # Detect quality with more aggressive thresholds
            quality = self._detect_quality(img_gray)
            print(f"  → Detected quality: {quality}")
            
            # Force aggressive for very poor images
            if quality in ['POOR', 'VERY_POOR']:
                aggressive = True
            
            # Try preprocessing
            if aggressive or quality != 'EXCELLENT':
                processed = self._preprocess_by_quality(img_gray, quality, aggressive)
                
                # If preprocessing made it worse, use simpler approach
                if self._is_worse(original_gray, processed):
                    print(f"  ⚠ Preprocessing made it worse, trying alternative...")
                    processed = self._alternative_preprocessing(img_gray)
                
                return Image.fromarray(processed)
            else:
                # Image is excellent, minimal processing
                return Image.fromarray(self._minimal_preprocessing(img_gray))
        
        except Exception as e:
            print(f"  ⚠ Preprocessing failed: {str(e)}")
            return image
    
    def _detect_quality(self, img: np.ndarray) -> str:
        """
        More aggressive quality detection
        Biased towards marking images as poor quality
        """
        std_dev = img.std()
        max_val = img.max()
        min_val = img.min()
        contrast_range = max_val - min_val
        mean_val = img.mean()
        
        # Sharpness
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # More aggressive scoring (biased towards poor)
        score = 0
        
        # Contrast scoring (stricter)
        if contrast_range > 220:
            score += 2
        elif contrast_range > 180:
            score += 1
        elif contrast_range < 120:  # Stricter threshold
            score -= 3  # More penalty
        elif contrast_range < 150:
            score -= 1
        
        # Std dev scoring (stricter)
        if std_dev > 70:
            score += 2
        elif std_dev > 50:
            score += 1
        elif std_dev < 35:  # Stricter threshold
            score -= 3  # More penalty
        elif std_dev < 45:
            score -= 1
        
        # Max value scoring (stricter)
        if max_val > 250:
            score += 2
        elif max_val > 235:
            score += 1
        elif max_val < 215:  # Stricter threshold
            score -= 2  # More penalty
        
        # Sharpness scoring
        if sharpness > 600:
            score += 2
        elif sharpness > 300:
            score += 1
        elif sharpness < 80:
            score -= 2
        
        # Debug output
        print(f"     Quality metrics: contrast={contrast_range}, std={std_dev:.1f}, max={max_val}, sharp={sharpness:.0f}, score={score}")
        
        # More conservative thresholds (bias towards poor)
        if score >= 5:
            return 'EXCELLENT'
        elif score >= 2:
            return 'GOOD'
        elif score >= -1:  # Stricter
            return 'MEDIUM'
        elif score >= -3:
            return 'POOR'
        else:
            return 'VERY_POOR'
    
    def _is_worse(self, original: np.ndarray, processed: np.ndarray) -> bool:
        """
        Check if preprocessing made the image worse
        Compare information content
        """
        try:
            # Count unique values (more is better)
            orig_unique = len(np.unique(original))
            proc_unique = len(np.unique(processed))
            
            # If processed has much fewer unique values, might be over-processed
            if proc_unique < orig_unique * 0.3:
                return True
            
            # Check if processed is mostly white or black (over-binarized)
            if processed.mean() > 250 or processed.mean() < 5:
                return True
            
            return False
        except:
            return False
    
    def _preprocess_by_quality(self, img: np.ndarray, quality: str, force_aggressive: bool) -> np.ndarray:
        """Apply appropriate preprocessing based on quality"""
        if force_aggressive or quality in ['POOR', 'VERY_POOR']:
            return self._aggressive_preprocessing(img)
        elif quality == 'MEDIUM':
            return self._enhanced_preprocessing(img)
        elif quality == 'GOOD':
            return self._standard_preprocessing(img)
        else:
            return self._minimal_preprocessing(img)
    
    def _limit_size(self, img: np.ndarray) -> np.ndarray:
        """Limit size but keep sufficient resolution"""
        height, width = img.shape[:2]
        
        if height > self.max_dimension or width > self.max_dimension:
            print(f"     Resizing from {width}x{height}", end=" ")
            scale = self.max_dimension / max(height, width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            print(f"to {new_width}x{new_height}")
        
        return img
    
    # ========================================================================
    # PREPROCESSING PIPELINES
    # ========================================================================
    
    def _minimal_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Minimal for excellent images"""
        print("  → Applying minimal preprocessing...")
        img = cv2.GaussianBlur(img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img
    
    def _standard_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Standard for good images"""
        print("  → Applying standard preprocessing...")
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 15, 8)
        return img
    
    def _enhanced_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Enhanced for medium quality"""
        print("  → Applying enhanced preprocessing...")
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = self._adjust_gamma(img, 1.3)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 19, 10)
        img = cv2.bitwise_and(otsu, adaptive)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img
    
    def _aggressive_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Aggressive for poor quality"""
        print("  → Applying AGGRESSIVE preprocessing...")
        
        # Upscale
        scale = min(2.0, self.max_dimension / max(img.shape))
        if scale > 1.0:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            print(f"     • Upscaling {scale}x")
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Extreme contrast
        print(f"     • Extreme contrast enhancement")
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        img = self._adjust_gamma(img, 1.5)
        
        # Remove background
        print(f"     • Removing background")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.subtract(background, img)
        img = cv2.bitwise_not(img)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Denoise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Triple binarization
        print(f"     • Triple binarization")
        _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_g = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 27, 15)
        adaptive_m = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 27, 15)
        img = cv2.bitwise_and(otsu, adaptive_g)
        img = cv2.bitwise_and(img, adaptive_m)
        
        # Cleanup
        print(f"     • Morphological cleanup")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # Sharpen
        gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        
        return img
    
    def _alternative_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """
        Alternative preprocessing if main one fails
        Simpler but sometimes more effective
        """
        print("  → Trying alternative (simpler) preprocessing...")
        
        # Just extreme contrast and simple threshold
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return img
    
    def _adjust_gamma(self, img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Gamma correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def upscale_image(self, image: Image.Image, target_dpi: int = 300) -> Image.Image:
        """Upscale to target DPI"""
        try:
            width, height = image.size
            current_dpi = image.info.get('dpi', (72, 72))
            if isinstance(current_dpi, tuple):
                current_dpi = current_dpi[0]
            
            scale_factor = target_dpi / current_dpi
            
            if 1 < scale_factor < 5:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                if new_width <= self.max_dimension and new_height <= self.max_dimension:
                    print(f"  → Upscaling from {width}x{height} to {new_width}x{new_height}")
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"  ⚠ Upscaling failed: {str(e)}")
        
        return image
    
    def quick_preprocess(self, image: Image.Image) -> Image.Image:
        """Emergency fallback"""
        try:
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_array = self._limit_size(img_array)
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return Image.fromarray(img_array)
        except Exception:
            return image