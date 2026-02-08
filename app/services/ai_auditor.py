# import os
# import torch
# from pathlib import Path
# from dotenv import load_dotenv
# from transformers import (
#     AutoConfig, 
#     AutoTokenizer, 
#     AutoModelForSeq2SeqLM, 
#     AutoModelForCausalLM
# )

# # Load variables from .env if present
# load_dotenv()

# class OfflineLCAuditor:
#     print("Initializing LC Auditor Model...", os.getenv("MODEL_NAME"))
#     # use this when not changing the directory to store the models
#     # def __init__(self, model_name="google/flan-t5-large"):
#     #     self.model_name = model_name

#     #     # Detect device
#     #     if torch.cuda.is_available():
#     #         self.device = torch.device("cuda")
#     #         torch_dtype = torch.float16   # Faster + less VRAM
#     #     else:
#     #         self.device = torch.device("cpu")
#     #         torch_dtype = torch.float32

#     #     print(f"[LCAuditor] Using device: {self.device}")

#     #     # Load tokenizer (default Hugging Face cache)
#     #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)

#     #     # Load model
#     #     self.model = AutoModelForSeq2SeqLM.from_pretrained(
#     #         model_name,
#     #         torch_dtype=torch_dtype
#     #     )

#     #     # Move model to device
#     #     self.model.to(self.device)
#     #     self.model.eval()
        
#     # use this when using a different directory to store the models
#     # def __init__(self, model_name="google/flan-t5-large", model_root=r"D:\AI_Models\huggingface"):
#     #     self.model_name = model_name
#     #     self.model_root = Path(model_root)

#     #     # Detect device
#     #     if torch.cuda.is_available():
#     #         self.device = torch.device("cuda")
#     #         dtype = torch.float16  # FP16 on GPU
#     #         print(f"[LCAuditor] GPU detected: {torch.cuda.get_device_name(0)}")
#     #     else:
#     #         self.device = torch.device("cpu")
#     #         dtype = torch.float32
#     #         print("[LCAuditor] Using CPU")

#     #     # Load tokenizer
#     #     self.tokenizer = AutoTokenizer.from_pretrained(
#     #         model_name,
#     #         cache_dir=self.model_root
#     #     )

#     #     # Load model
#     #     self.model = AutoModelForSeq2SeqLM.from_pretrained(
#     #         model_name,
#     #         cache_dir=self.model_root,
#     #         device_map="auto",  # Automatically maps model to GPU if available
#     #         torch_dtype=dtype
#     #     )

#     #     self.model.eval()

#     def __init__(self, model_name=None, model_root=None):
#         # 1️⃣ Resolve Model Name
#         self.model_name = os.getenv("MODEL_NAME", model_name or "google/flan-t5-large")
#         print(f"[LCAuditor] Resolved model name: {self.model_name}")

#         # 2️⃣ Resolve Model Root (optional local folder)
#         env_root = os.getenv("MODEL_PATH", model_root)
#         if env_root and str(env_root).strip():
#             self.model_root = Path(env_root)
#             print(f"[LCAuditor] Using local model root: {self.model_root}")
#         else:
#             self.model_root = None
#             print(f"[LCAuditor] No MODEL_PATH specified — using default Hugging Face cache")

#         # 3️⃣ Detect Device
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#         print(f"[LCAuditor] Device: {self.device} | Path: {self.model_root or 'Default Cache'}")

#         # 4️⃣ Build loader args
#         loader_args = {"trust_remote_code": True, "revision": "main"}
#         hf_token = os.getenv("HF_TOKEN")
#         if hf_token:
#             os.environ["HF_TOKEN"] = hf_token
#         if self.model_root:
#             loader_args["cache_dir"] = str(self.model_root)

#         # 5️⃣ Determine local-only
#         # If no model_root specified, allow downloading from HF
#         local_only = bool(self.model_root and (Path(self.model_root)/self.model_name).exists())
#         print(f"[LCAuditor] local_only={local_only}")

#         # 6️⃣ Protect GPTQ / 72B models from CPU-only load
#         if "72B" in self.model_name and not torch.cuda.is_available():
#             raise RuntimeError(
#                 "[LCAuditor] 72B GPTQ model requires CUDA GPU — cannot load on CPU"
#             )

#         # 7️⃣ Load Tokenizer & Config
#         try:
#             print(f"[LCAuditor] Loading tokenizer/config for {self.model_name}...")
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 self.model_name,
#                 local_files_only=local_only,
#                 **loader_args
#             )
#             config = AutoConfig.from_pretrained(
#                 self.model_name,
#                 local_files_only=local_only,
#                 **loader_args
#             )
#             print("[LCAuditor] Tokenizer & config loaded successfully")
#         except Exception as e:
#             raise RuntimeError(f"[LCAuditor] Failed to load tokenizer/config: {e}")

#         # 8️⃣ Load Model (Seq2Seq vs Causal)
#         try:
#             if config.is_encoder_decoder:
#                 print(f"[LCAuditor] Loading Seq2Seq model: {self.model_name}")
#                 self.model = AutoModelForSeq2SeqLM.from_pretrained(
#                     self.model_name,
#                     torch_dtype=self.dtype,
#                     local_files_only=local_only,
#                     **loader_args
#                 ).to(self.device)
#             else:
#                 print(f"[LCAuditor] Loading Causal model: {self.model_name}")
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     self.model_name,
#                     torch_dtype=self.dtype,
#                     device_map="auto" if torch.cuda.is_available() else None,
#                     local_files_only=local_only,
#                     **loader_args
#                 )
#             self.model.eval()
#             print("[LCAuditor] Model loaded successfully")
#         except Exception as e:
#             raise RuntimeError(f"[LCAuditor] Failed to load model: {e}")
    
    
#     def _prepare_inputs(self, prompt: str, max_length: int = 1024):
#         """
#         Tokenize inputs and move tensors to the correct device.
#         """
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=max_length
#         )
#         # Move each tensor to the correct device
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
#         return inputs

#     def _run_inference(self, prompt: str, max_tokens: int = 256) -> str:
#         """
#         Runs model inference safely on GPU or CPU.
#         """
#         inputs = self._prepare_inputs(prompt)

#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             do_sample=False,
#             num_beams=4,
#             temperature=0.3,
#             repetition_penalty=1.2,
#             length_penalty=1.0,
#             early_stopping=True
#         )

#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#         return decoded
    
#     def generate_merged_text(self, original_text: str, instruction: str) -> str:
#         """
#         Intelligently applies amendment instructions to original text.
#         Handles DELETE/REPLACE, INSTEAD OF, and TO READ AS patterns.
#         """
#         # Step 1: Aggressively desquash the instruction
#         instruction = self._desquash_instruction(instruction)
        
#         # Step 2: Try rule-based extraction first (faster and more reliable)
#         result = self._try_rule_based_merge(original_text, instruction)
#         if result:
#             return result
        
#         # Step 3: Fall back to AI for complex cases
#         return self._ai_merge(original_text, instruction)

#     def _desquash_instruction(self, text: str) -> str:
#         """
#         Aggressively separates squashed banking keywords.
#         Example: "CLAUENO.27TOREADAS" -> "CLAUSE NO.27 TO READ AS"
#         """
#         # Replace common patterns
#         text = text.replace("''", "'").replace('"', "'")
        
#         # Banking-specific desquashing
#         replacements = {
#             'CLAUENO': 'CLAUSE NO',
#             'CLAUSENO': 'CLAUSE NO',
#             'TOREADAS': 'TO READ AS',
#             'TOREAD': 'TO READ',
#             'INSTEADOF': 'INSTEAD OF',
#             'REPLACEBY': 'REPLACE BY',
#             'FIELDNO': 'FIELD NO',
#             'FIELD46A': 'FIELD 46A',
#             'FIELD47A': 'FIELD 47A',
#         }
        
#         for old, new in replacements.items():
#             text = re.sub(old, new, text, flags=re.IGNORECASE)
        
#         # Insert spaces between keywords and adjacent text
#         keywords = ['CLAUSE', 'NO', 'TO', 'READ', 'AS', 'INSTEAD', 'OF', 'DELETE', 
#                    'REPLACE', 'BY', 'WITH', 'FIELD', 'NOW']
        
#         for kw in keywords:
#             # Add space after keyword if followed by alphanumeric
#             text = re.sub(f'({kw})([A-Z0-9])', r'\1 \2', text, flags=re.IGNORECASE)
#             # Add space before keyword if preceded by alphanumeric
#             text = re.sub(f'([A-Z0-9])({kw})', r'\1 \2', text, flags=re.IGNORECASE)
        
#         # Clean up multiple spaces
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         return text

#     def _try_rule_based_merge(self, original_text: str, instruction: str) -> str:
#         """
#         Attempts to apply the instruction using regex patterns.
#         Returns None if no pattern matches.
#         """
#         instruction_upper = instruction.upper()
        
#         # Pattern 1: "TO READ AS 'X'" - Complete replacement
#         match = re.search(r"TO\s+READ\s+AS\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
#         if match:
#             return match.group(1).strip()
        
#         # Pattern 2: "'X' INSTEAD OF 'Y'" - Substitution
#         match = re.search(r"['\"](.+?)['\"]\s+INSTEAD\s+OF\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
#         if match:
#             new_val, old_val = match.group(1).strip(), match.group(2).strip()
#             if old_val.upper() in original_text.upper():
#                 # Case-insensitive replacement
#                 return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
#         # Pattern 3: "DELETE 'X' REPLACE BY 'Y'" - Substitution
#         match = re.search(r"DELETE\s+['\"](.+?)['\"]\s+REPLACE\s+(?:BY|WITH)\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
#         if match:
#             old_val, new_val = match.group(1).strip(), match.group(2).strip()
#             if old_val.upper() in original_text.upper():
#                 return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
#         # Pattern 4: No quotes, but DELETE X REPLACE BY Y (looser pattern)
#         match = re.search(r"DELETE\s+(.+?)\s+REPLACE\s+(?:BY|WITH)\s+(.+?)(?:\s|$)", instruction, re.IGNORECASE)
#         if match:
#             old_val = match.group(1).strip().strip("'\"")
#             new_val = match.group(2).strip().strip("'\"")
#             if old_val.upper() in original_text.upper():
#                 return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
#         return None

#     def _ai_merge(self, original_text: str, instruction: str) -> str:
#         """
#         Uses AI to merge when rule-based patterns fail.
#         """
#         prompt = self._build_merge_prompt(original_text, instruction)
#         return self._run_inference(prompt, max_tokens=256)

#     def _build_merge_prompt(self, original_text: str, instruction: str) -> str:
#         """
#         Constructs a precise prompt for the AI model.
#         """
#         return f"""Task: Update the original clause by applying the change instruction.

# Rules:
# 1. If instruction says "DELETE X REPLACE BY Y", substitute X with Y in the original
# 2. If instruction says "X INSTEAD OF Y", substitute Y with X in the original
# 3. If instruction says "TO READ AS X", replace entire clause with X
# 4. Output ONLY the updated clause text
# 5. Do NOT include "CLAUSE NO" or instruction keywords in output

# Examples:

# Example 1:
# Original: Documents presented within 30 days from vessel notice.
# Instruction: DELETE '30 DAYS' REPLACE BY '45 DAYS'
# Output: Documents presented within 45 days from vessel notice.

# Example 2:
# Original: Insurance covered by National Insurance Company.
# Instruction: DELETE '5 DAYS BEFORE' REPLACE BY 'AT LEAST 3 DAYS BEFORE'
# Output: Insurance advice at least 3 days before shipment.

# Example 3:
# Original: Independent inspector to be nominated.
# Instruction: TO READ AS 'M/S. SAWANT AND CO. PRIVATE LTD'
# Output: M/S. SAWANT AND CO. PRIVATE LTD

# Now apply this logic:

# Original Clause: {original_text}
# Change Instruction: {instruction}
# Output:"""

#     def _run_inference(self, prompt: str, max_tokens: int = 256) -> str:
#         """
#         Runs inference with the model.
#         """
#         inputs = self.tokenizer(
#             prompt, 
#             return_tensors="pt", 
#             truncation=True, 
#             max_length=1024
#         ).to(self.device)
        
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=max_tokens,
#             do_sample=False,
#             num_beams=4,
#             temperature=0.3,
#             repetition_penalty=1.2,  # Prevent repetition
#             length_penalty=1.0,
#             early_stopping=True
#         )
        
#         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
#         # Clean up the output
#         decoded = decoded.split("Output:")[-1].strip()
#         decoded = re.sub(r'^(CLAUSE|ITEM)\s*(?:NO\.)?\s*\d+\s*', '', decoded, flags=re.IGNORECASE)
#         decoded = re.sub(r'^(?:NOW\s+)?TO\s+READ\s+AS\s*', '', decoded, flags=re.IGNORECASE)
        
#         return decoded.strip().strip("'\"")

#     def verify_clause(self, requirement_text: str, document_context: str) -> str:
#         """
#         Verifies if a clause meets requirements (for audit purposes).
#         """
#         prompt = f"""Check if the requirement is met by the document context.

# Requirement: {requirement_text}
# Document Context: {document_context}

# Answer with 'YES' if met, 'NO' if not met, or 'PARTIAL' if partially met.
# Answer:"""
        
#         return self._run_inference(prompt, max_tokens=50)

import os
import re
import torch
from pathlib import Path
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM
)

class OfflineLCAuditor:
    def __init__(self, model_name=None, model_root=None):
        # 1️⃣ Resolve Model Name & Paths
        self.model_name = os.getenv("MODEL_NAME", model_name or "google/flan-t5-large")
        env_root = os.getenv("MODEL_PATH", model_root or r"D:\AI_Models\huggingface")
        self.model_root = Path(env_root) if env_root else None
        
        # 2️⃣ Detect Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # 3️⃣ Build loader args
        loader_args = {"trust_remote_code": True}
        if self.model_root:
            loader_args["cache_dir"] = str(self.model_root)

        # Determine if we should stay offline
        # It's local_only if the model folder exists in your D: drive
        local_only = bool(self.model_root and (self.model_root/self.model_name.replace("/", "--")).exists())
        
        print(f"[*] Loading {self.model_name} | Device: {self.device} | Local Only: {local_only}")

        # 4️⃣ Load Tokenizer & Config
        try:
            # FIX: Added use_fast=False to solve the SentencePiece/Tiktoken error
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=local_only,
                use_fast=False, 
                **loader_args
            )
            self.config = AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=local_only,
                **loader_args
            )
        except Exception as e:
            raise RuntimeError(f"[LCAuditor] Tokenizer/Config Load Failed: {e}")

        # 5️⃣ Load Model (restoring your Seq2Seq vs Causal logic)
        try:
            if self.config.is_encoder_decoder:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    local_files_only=local_only,
                    **loader_args
                ).to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=local_only,
                    **loader_args
                )
            self.model.eval()
            print("✓ AI Engine initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"[LCAuditor] Model Load Failed: {e}")

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """The standard generation method."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=2,
                temperature=0.1
            )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return re.sub(r'^Output:\s*', '', decoded, flags=re.IGNORECASE)

    def generate_merged_text(self, original_text: str, instruction: str) -> str:
        """Restores your specialized merging logic with banking desquashing."""
        instruction = self._desquash_instruction(instruction)
        
        # 1. Regex Try
        regex_res = self._try_rule_based_merge(original_text, instruction)
        if regex_res: return regex_res
        
        # 2. AI Fallback
        prompt = f"""Task: Update the original LC clause based on the instruction.
Original: {original_text}
Instruction: {instruction}
Final Clause Text:"""
        return self.generate(prompt)

    def _desquash_instruction(self, text: str) -> str:
        replacements = {'CLAUENO': 'CLAUSE NO', 'TOREADAS': 'TO READ AS', 'INSTEADOF': 'INSTEAD OF'}
        for old, new in replacements.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)
        return text.strip()

    def _try_rule_based_merge(self, original_text: str, instruction: str) -> str:
        # Regex Pattern for: DELETE 'X' REPLACE BY 'Y'
        match = re.search(r"DELETE\s+['\"](.+?)['\"]\s+REPLACE\s+(?:BY|WITH)\s+['\"](.+?)['\"]", instruction, re.IGNORECASE)
        if match:
            old_val, new_val = match.group(1).strip(), match.group(2).strip()
            if old_val.upper() in original_text.upper():
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        return None