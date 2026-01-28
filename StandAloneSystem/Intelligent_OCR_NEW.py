"""
Enhanced OCR Module with Intelligent Auto-Denoising
Optimized for blurry, low-quality, and challenging documents
Supports Tesseract, EasyOCR, and PaddleOCR with automatic selection
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import tempfile


# =========================
# IMAGE PREPROCESSOR
# =========================
class ImagePreprocessor:
    def __init__(self):
        self.default_dpi = 300

    # ---------- Quality Analysis ----------
    def analyze_quality(self, img: np.ndarray) -> dict:
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        noise_score = np.std(img)
        return {
            "blur": blur_score,
            "noise": noise_score
        }

    def determine_level(self, metrics: dict) -> str:
        if metrics["blur"] < 80 or metrics["noise"] > 35:
            return "heavy"
        elif metrics["blur"] < 200:
            return "medium"
        return "light"

    # ---------- Main Pipeline ----------
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        img = np.array(image)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        metrics = self.analyze_quality(img)
        level = self.determine_level(metrics)

        img = self._denoise(img, level)
        img = self._enhance_contrast(img)
        img = self._sharpen(img, level)
        img = self._binarize(img, level)
        img = self._deskew(img)
        img = self._remove_borders(img)

        return Image.fromarray(img)

    # ---------- Steps ----------
    def _denoise(self, img, level):
        if level == "heavy":
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.fastNlMeansDenoising(img, None, h=20)
        elif level == "medium":
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.fastNlMeansDenoising(img, None, h=10)
        img = cv2.bilateralFilter(img, 9, 75, 75)
        return img

    def _enhance_contrast(self, img):
        clahe = cv2.createCLAHE(2.0, (8, 8))
        return clahe.apply(img)

    def _sharpen(self, img, level):
        if level == "heavy":
            kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        else:
            kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        return cv2.filter2D(img, -1, kernel)

    def _binarize(self, img, level):
        if level == "heavy":
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        return img

    def _deskew(self, img):
        coords = np.column_stack(np.where(img > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.5:
            return img
        (h, w) = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _remove_borders(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img[y:y+h, x:x+w]

    def upscale_image(self, image: Image.Image):
        dpi = image.info.get("dpi", (72,))[0]
        scale = self.default_dpi / dpi
        if scale > 1:
            w, h = image.size
            image = image.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
        return image


# =========================
# OCR PROCESSOR
# =========================
class EnhancedOCRProcessor:
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        import pytesseract, easyocr
        from paddleocr import PaddleOCR

        self.tesseract = pytesseract
        self.easyocr = easyocr.Reader(['en'], gpu=False)
        self.paddle = PaddleOCR(use_angle_cls=True, lang='en')

    def extract_text_from_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        img = self.preprocessor.upscale_image(img)
        processed = self.preprocessor.preprocess_for_ocr(img)

        img_np = np.array(processed)
        quality = self.preprocessor.analyze_quality(img_np)
        level = self.preprocessor.determine_level(quality)

        if level == "heavy":
            return "\n".join(self.easyocr.readtext(img_np, detail=0))
        elif level == "medium":
            result = self.paddle.ocr(img_np, cls=True)
            return "\n".join([l[1][0] for l in result[0]]) if result else ""
        else:
            return self.tesseract.image_to_string(processed, config="--oem 3 --psm 6")


# =========================
# PDF PROCESSOR
# =========================
class PDFProcessorEnhanced:
    def __init__(self):
        self.ocr = EnhancedOCRProcessor()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for p in pdf.pages:
                    if p.extract_text():
                        text += p.extract_text() + "\n"
            if text.strip():
                return text
        except:
            pass

        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, dpi=300)
        output = ""
        for i, page in enumerate(pages):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name)
                output += f"\n--- Page {i+1} ---\n"
                output += self.ocr.extract_text_from_image(tmp.name)
                os.remove(tmp.name)
        return output


# =========================
# DOCUMENT PROCESSOR
# =========================
class DocumentProcessor:
    def __init__(self):
        self.pdf = PDFProcessorEnhanced()
        self.ocr = EnhancedOCRProcessor()

    def process_document(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self.pdf.extract_text_from_pdf(file_path)
        elif ext in [".png",".jpg",".jpeg",".bmp",".tiff",".gif"]:
            return self.ocr.extract_text_from_image(file_path)
        elif ext in [".txt",".text"]:
            return Path(file_path).read_text(errors="ignore")
        else:
            return ""


if __name__ == "__main__":
    print("Enhanced OCR Processor – AUTO denoising, AUTO engine selection")
