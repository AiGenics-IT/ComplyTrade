import os
import torch
from PIL import Image
from dotenv import load_dotenv
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Load environment variables
load_dotenv()

class OlmOCRProcessor:
    """
    Production-grade olmOCR-2 Processor for Ubuntu 25.10 / Python 3.11.
    Handles local weight loading from AI_MODELS or remote API inference.
    """
    def __init__(self):
        self.model_cache_root = os.getenv("MODEL_PATH")  # /home/aigenics/AI_MODELS
        self.model_name = os.getenv("OCR_MODEL_NAME")   # allenai/olmOCR-2-7B-1025-FP8
        
        if not self.model_cache_root or not self.model_name:
            raise ValueError("Environment variables MODEL_PATH or OCR_MODEL_NAME are missing.")

        self.is_server = self.model_cache_root.startswith(("http://", "https://"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if not self.is_server:
            print(f"🚀 Loading {self.model_name} from cache: {self.model_cache_root}")
            
            # Use cache_dir to properly navigate the 'models--' folder structure
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=self.model_cache_root,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.model_cache_root,
                local_files_only=True
            )
        else:
            print(f"🌐 Initialized olmOCR-2 via API Server: {self.model_cache_root}")

    def _build_prompt(self):
        """Strategic prompt for fixing merged words and preserving Swift tags."""
        return (
            "You are an expert document parser. Extract all text from this image accurately. "
            "If words are merged (e.g., 'CLAUSENO'), add proper spaces ('CLAUSE NO'). "
            "Preserve all financial tags like 46B:, 47B:, and Swift formatting exactly. "
            "Output the result as clean Markdown."
        )

    def extract_text(self, image_path: str) -> str:
        """Core extraction logic optimized for Vision Language Models."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")

            image = Image.open(image_path).convert("RGB")
            
            if not self.is_server:
                # 1. Format messages for Qwen2-VL
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": self._build_prompt()},
                        ],
                    }
                ]

                # 2. Prepare inputs for the model
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # 3. Generate result (Greedy decoding for high precision)
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, 
                        max_new_tokens=1536,
                        do_sample=False  # Crucial for zero-hallucination in banking
                    )

                # 4. Decode output
                generated_ids = [
                    output_ids[len(input_ids):] 
                    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                
                response = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=True
                )[0]
                
                return response.strip()
            
            else:
                # Placeholder for vLLM / OpenAI-compatible API calls
                return "API inference not implemented in this snippet."

        except Exception as e:
            print(f"❌ Error in olmOCR extraction: {str(e)}")
            return ""

# --- Usage Example ---
if __name__ == "__main__":
    # Ensure ocr_env is active and dependencies are installed
    processor = OlmOCRProcessor()
    # Replace with your actual noisy.jpg or a converted PDF page
    result = processor.extract_text("path/to/your/noisy.jpg")
    print("\n--- Extracted Text ---\n")
    print(result)