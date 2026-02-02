"""
Complete OCR Solution for Poor Quality Documents
Combines aggressive preprocessing with optimized Tesseract settings
"""

import numpy as np
from PIL import Image
import cv2


def preprocess_poor_quality_image(image_path: str, output_path: str = None) -> Image.Image:
    """
    Preprocess extremely poor quality images
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save preprocessed image
    
    Returns:
        Preprocessed PIL Image
    """
    print(f"\n{'='*80}")
    print(f"PREPROCESSING POOR QUALITY IMAGE")
    print(f"{'='*80}")
    
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    print(f"Original: {img_array.shape[1]}x{img_array.shape[0]}")
    print(f"  Min/Max: {img_array.min()}/{img_array.max()}")
    print(f"  Mean: {img_array.mean():.1f}")
    
    # STEP 1: Upscale first (better quality)
    scale_factor = 2.0  # Double the size
    new_width = int(img_array.shape[1] * scale_factor)
    new_height = int(img_array.shape[0] * scale_factor)
    img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    print(f"\nStep 1: Upscaled to {new_width}x{new_height}")
    
    # STEP 2: Extreme contrast enhancement
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    img_array = clahe.apply(img_array)
    print(f"Step 2: Enhanced contrast")
    
    # STEP 3: Denoise
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    print(f"Step 3: Denoised")
    
    # STEP 4: Remove background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    background = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    img_array = cv2.subtract(background, img_array)
    img_array = cv2.bitwise_not(img_array)
    img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX)
    print(f"Step 4: Removed background")
    
    # STEP 5: Aggressive binarization
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Step 5: Binarized")
    
    # STEP 6: Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    print(f"Step 6: Cleaned up")
    
    print(f"\nFinal: {img_array.shape[1]}x{img_array.shape[0]}")
    print(f"  Min/Max: {img_array.min()}/{img_array.max()}")
    print(f"  Unique values: {len(np.unique(img_array))}")
    print(f"{'='*80}\n")
    
    result = Image.fromarray(img_array)
    
    if output_path:
        result.save(output_path)
        print(f"✓ Saved preprocessed image to: {output_path}")
    
    return result


def extract_text_with_best_settings(image_path: str, save_preprocessed: bool = False) -> str:
    """
    Extract text from poor quality image using best settings
    
    Args:
        image_path: Path to image
        save_preprocessed: Save the preprocessed image for inspection
    
    Returns:
        Extracted text
    """
    import pytesseract
    
    # Preprocess
    preprocessed_path = image_path.replace('.jpg', '_preprocessed.png') if save_preprocessed else None
    preprocessed = preprocess_poor_quality_image(image_path, preprocessed_path)
    
    print("Extracting text with Tesseract...")
    print("  Trying multiple configurations...\n")
    
    # Try multiple Tesseract configurations
    configs = [
        # Config 1: Default with best quality
        '--oem 1 --psm 6',
        
        # Config 2: Assume single uniform block
        '--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:,/-()$%& ',
        
        # Config 3: Treat as single column
        '--oem 1 --psm 4',
        
        # Config 4: Sparse text
        '--oem 1 --psm 11',
    ]
    
    best_text = ""
    best_confidence = 0
    
    for i, config in enumerate(configs, 1):
        try:
            print(f"  Config {i}: {config[:30]}...")
            
            # Get text and confidence
            data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(preprocessed, config=config)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            print(f"    Confidence: {avg_conf:.1f}% | Length: {len(text)} chars")
            
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_text = text
        
        except Exception as e:
            print(f"    Failed: {str(e)}")
            continue
    
    print(f"\n  ✓ Best result: {best_confidence:.1f}% confidence, {len(best_text)} chars")
    
    return best_text


# CLI tool for testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ocr_poor_quality.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Extract text
    text = extract_text_with_best_settings(image_path, save_preprocessed=True)
    
    # Save result
    output_path = image_path.replace('.jpg', '_extracted.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Output saved to: {output_path}")
    print(f"\nFirst 500 characters:")
    print(text[:500])