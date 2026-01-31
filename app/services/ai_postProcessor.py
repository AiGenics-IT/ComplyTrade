# Hybrid Regix first post processor with selective AI cleanup
import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class AIOCRPostProcessor:
    """
    Super-fast Hybrid Processor.
    Regex handles the structure (instant).
    AI handles only the messy narrative (selective).
    """
    
    # use when not defining custom folder
    # def __init__(self, model_name="google/flan-t5-base"):
    #     self.model_name = model_name

    #     # Detect device
    #     if torch.cuda.is_available():
    #         self.device = torch.device("cuda")
    #         torch_dtype = torch.float16  # Faster + less VRAM
    #     else:
    #         self.device = torch.device("cpu")
    #         torch_dtype = torch.float32

    #     print(f"[MyT5Model] Using device: {self.device}")

    #     # Load tokenizer
    #     self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    #     # Load model
    #     self.model = T5ForConditionalGeneration.from_pretrained(
    #         model_name,
    #         torch_dtype=torch_dtype
    #     )

    #     # Move model to device
    #     self.model.to(self.device)
    #     self.model.eval()

    # use when defining custom folder
    def __init__(
        self,
        model_name="google/flan-t5-base",
        model_root=r"D:\AI_Models\huggingface"
    ):
        self.model_name = model_name
        self.model_root = Path(model_root)

        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch_dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            torch_dtype = torch.float32

        print(f"[MyT5Model] Using device: {self.device}")

        # Load tokenizer from custom directory
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_root / model_name
        )

        # Load model from custom directory
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_root / model_name,
            torch_dtype=torch_dtype
        )

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

    def _ai_fix_narrative(self, text: str) -> str:
        """Only called for messy parts. Low max_length for speed."""
        # Check if it actually needs fixing (contains squashed words)
        if len(text) < 10 or " " in text: # If it already has spaces, skip AI
            return text

        prompt = f"add spaces to this banking text: {text}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad(): # Disable gradient calculation for speed
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100, # Keep it short
                do_sample=False     # Greedy search is faster than beam search
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def clean_text(self, text: str) -> str:
        # 1. FAST REGEX PRE-PROCESS (Instant)
        # Squash tags and Message types immediately
        text = re.sub(r'Message\s*type\s*:?\s*(\d{3})', r'Messagetype:\1', text, flags=re.IGNORECASE)
        text = re.sub(r'(\d{2,3})\s*([A-Z])?\s*:', r'\1\2:', text)

        # 2. SELECTIVE AI (Only process long strings without spaces)
        lines = text.splitlines()
        final_lines = []
        
        for line in lines:
            # If line is a tag (e.g., 47A:...) and the content is squashed
            if ":" in line:
                tag, content = line.split(":", 1)
                # If content is long and has very few spaces, it's probably squashed
                if len(content) > 15 and content.count(' ') < 2:
                    content = self._ai_fix_narrative(content.strip())
                final_lines.append(f"{tag.strip()}:{content}")
            else:
                final_lines.append(line)

        # 3. FAST REGEX POST-PROCESS
        result = "\n".join(final_lines)
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result) # CamelCase fix
        return result



import re

class AIOCRPostProcessor:
    """
    Intelligent Space Insertion Processor.
    Does NOT modify characters or add labels. 
    Only inserts spaces into squashed banking terminology.
    """
    def __init__(self):
        # High-priority banking terms to identify split points
        self.keywords = [
            "CLAUSE", "FIELD", "LETTER", "AUTHORITY", "SIGNING", "BILL", "LADING",
            "DOCUMENTARY", "EVIDENCE", "REQUIRED", "AGAINST", "READ", "AS", 
            "AMENDMENT", "REFERENCE", "MESSAGE", "TYPE", "INSTEAD", "TOTAL",
            "SEQUENCE", "ACCEPTABLE", "COMPLY", "ORIGINAL", "INVOICE"
        ]
        
    def _insert_spaces(self, text: str) -> str:
        """
        Walks through squashed text and inserts spaces based on keywords
        without changing the original character sequence.
        """
        if not text or len(text) < 5:
            return text
            
        result = text
        for word in self.keywords:
            # Matches the word only if it's NOT already surrounded by spaces
            # and is part of a larger alphanumeric string
            pattern = re.compile(f"(?<=[A-Za-z0-9])({word})(?=[A-Za-z0-9])", re.IGNORECASE)
            result = pattern.sub(r' \1 ', result)
            
            # Catch cases where word is at the start/end of a squash
            # e.g., "TOREADAS" -> "TO READ AS"
            result = re.sub(f"([A-Z])({word})", r"\1 \2", result, flags=re.IGNORECASE)
            result = re.sub(f"({word})([A-Z])", r"\1 \2", result, flags=re.IGNORECASE)

        # Fix double spaces created by the logic
        return re.sub(r' +', ' ', result).strip()

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        lines = text.splitlines()
        processed_lines = []

        for line in lines:
            # 1. Skip lines that are already well-formatted (contains multiple spaces)
            if line.count(' ') > 5:
                processed_lines.append(line)
                continue
            
            # 2. Handle lines with SWIFT tags (preserve the tag, space the content)
            if ':' in line and len(line.split(':')[0]) < 5:
                tag, content = line.split(':', 1)
                # Keep tag squashed (e.g. 47B:), fix the rest
                spaced_content = self._insert_spaces(content)
                processed_lines.append(f"{tag.strip()}:{spaced_content}")
            
            # 3. Handle purely squashed lines (Field continuations)
            else:
                processed_lines.append(self._insert_spaces(line))

        # Join and do a final pass for specific common squashes like "L/C" or "B/L"
        final_text = "\n".join(processed_lines)
        
        # Ensure tags are always at the start of a line for your Extractor
        final_text = re.sub(r'(\d{2,3}[A-Z]?:)', r'\n\1', final_text)
        
        return final_text.strip()