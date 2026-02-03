import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

class QwenLocalOCR:
    """
    OCR via local Qwen3-72B multimodal model.
    Can be used as a backend for EnhancedOCRProcessor.
    """

    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[QwenLocalOCR] Using device: {self.device}")

        # Load processor and model from local directory
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            local_files_only=True
        )
        self.model.eval()
        print(f"[QwenLocalOCR] Loaded model from {model_path}")

    def extract_text(self, image_path: str) -> str:
        """
        Perform OCR on the given image path and return extracted text
        """
        image = Image.open(image_path).convert("RGB")

        # Convert image to model input
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Prompt for OCR
        prompt = "Please extract all text from this image."

        # Tokenize prompt
        input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        text = self.processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
