from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re

class OfflineLCAuditor:
    def __init__(self, model_name="google/flan-t5-base"):  # Using base model for better results
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def generate_merged_text(self, original_text: str, instruction: str) -> str:
        """
        Intelligently applies amendment instructions to original text.
        Handles DELETE/REPLACE, INSTEAD OF, and TO READ AS patterns.
        """
        # Step 1: Aggressively desquash the instruction
        instruction = self._desquash_instruction(instruction)
        
        # Step 2: Try rule-based extraction first (faster and more reliable)
        result = self._try_rule_based_merge(original_text, instruction)
        if result:
            return result
        
        # Step 3: Fall back to AI for complex cases
        return self._ai_merge(original_text, instruction)

    def _desquash_instruction(self, text: str) -> str:
        """
        Aggressively separates squashed banking keywords.
        Example: "CLAUENO.27TOREADAS" -> "CLAUSE NO.27 TO READ AS"
        """
        # Replace common patterns
        text = text.replace("''", "'").replace('"', "'")
        
        # Banking-specific desquashing
        replacements = {
            'CLAUENO': 'CLAUSE NO',
            'CLAUSENO': 'CLAUSE NO',
            'TOREADAS': 'TO READ AS',
            'TOREAD': 'TO READ',
            'INSTEADOF': 'INSTEAD OF',
            'REPLACEBY': 'REPLACE BY',
            'FIELDNO': 'FIELD NO',
            'FIELD46A': 'FIELD 46A',
            'FIELD47A': 'FIELD 47A',
        }
        
        for old, new in replacements.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)
        
        # Insert spaces between keywords and adjacent text
        keywords = ['CLAUSE', 'NO', 'TO', 'READ', 'AS', 'INSTEAD', 'OF', 'DELETE', 
                   'REPLACE', 'BY', 'WITH', 'FIELD', 'NOW']
        
        for kw in keywords:
            # Add space after keyword if followed by alphanumeric
            text = re.sub(f'({kw})([A-Z0-9])', r'\1 \2', text, flags=re.IGNORECASE)
            # Add space before keyword if preceded by alphanumeric
            text = re.sub(f'([A-Z0-9])({kw})', r'\1 \2', text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _try_rule_based_merge(self, original_text: str, instruction: str) -> str:
        """
        Attempts to apply the instruction using regex patterns.
        Returns None if no pattern matches.
        """
        instruction_upper = instruction.upper()
        
        # Pattern 1: "TO READ AS 'X'" - Complete replacement
        match = re.search(r"TO\s+READ\s+AS\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: "'X' INSTEAD OF 'Y'" - Substitution
        match = re.search(r"['\"](.+?)['\"]\s+INSTEAD\s+OF\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
        if match:
            new_val, old_val = match.group(1).strip(), match.group(2).strip()
            if old_val.upper() in original_text.upper():
                # Case-insensitive replacement
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
        # Pattern 3: "DELETE 'X' REPLACE BY 'Y'" - Substitution
        match = re.search(r"DELETE\s+['\"](.+?)['\"]\s+REPLACE\s+(?:BY|WITH)\s+['\"](.+?)['\"]", instruction, re.IGNORECASE | re.DOTALL)
        if match:
            old_val, new_val = match.group(1).strip(), match.group(2).strip()
            if old_val.upper() in original_text.upper():
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
        # Pattern 4: No quotes, but DELETE X REPLACE BY Y (looser pattern)
        match = re.search(r"DELETE\s+(.+?)\s+REPLACE\s+(?:BY|WITH)\s+(.+?)(?:\s|$)", instruction, re.IGNORECASE)
        if match:
            old_val = match.group(1).strip().strip("'\"")
            new_val = match.group(2).strip().strip("'\"")
            if old_val.upper() in original_text.upper():
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
        return None

    def _ai_merge(self, original_text: str, instruction: str) -> str:
        """
        Uses AI to merge when rule-based patterns fail.
        """
        prompt = self._build_merge_prompt(original_text, instruction)
        return self._run_inference(prompt, max_tokens=256)

    def _build_merge_prompt(self, original_text: str, instruction: str) -> str:
        """
        Constructs a precise prompt for the AI model.
        """
        return f"""Task: Update the original clause by applying the change instruction.

Rules:
1. If instruction says "DELETE X REPLACE BY Y", substitute X with Y in the original
2. If instruction says "X INSTEAD OF Y", substitute Y with X in the original
3. If instruction says "TO READ AS X", replace entire clause with X
4. Output ONLY the updated clause text
5. Do NOT include "CLAUSE NO" or instruction keywords in output

Examples:

Example 1:
Original: Documents presented within 30 days from vessel notice.
Instruction: DELETE '30 DAYS' REPLACE BY '45 DAYS'
Output: Documents presented within 45 days from vessel notice.

Example 2:
Original: Insurance covered by National Insurance Company.
Instruction: DELETE '5 DAYS BEFORE' REPLACE BY 'AT LEAST 3 DAYS BEFORE'
Output: Insurance advice at least 3 days before shipment.

Example 3:
Original: Independent inspector to be nominated.
Instruction: TO READ AS 'M/S. SAWANT AND CO. PRIVATE LTD'
Output: M/S. SAWANT AND CO. PRIVATE LTD

Now apply this logic:

Original Clause: {original_text}
Change Instruction: {instruction}
Output:"""

    def _run_inference(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Runs inference with the model.
        """
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=4,
            temperature=0.3,
            repetition_penalty=1.2,  # Prevent repetition
            length_penalty=1.0,
            early_stopping=True
        )
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Clean up the output
        decoded = decoded.split("Output:")[-1].strip()
        decoded = re.sub(r'^(CLAUSE|ITEM)\s*(?:NO\.)?\s*\d+\s*', '', decoded, flags=re.IGNORECASE)
        decoded = re.sub(r'^(?:NOW\s+)?TO\s+READ\s+AS\s*', '', decoded, flags=re.IGNORECASE)
        
        return decoded.strip().strip("'\"")

    def verify_clause(self, requirement_text: str, document_context: str) -> str:
        """
        Verifies if a clause meets requirements (for audit purposes).
        """
        prompt = f"""Check if the requirement is met by the document context.

Requirement: {requirement_text}
Document Context: {document_context}

Answer with 'YES' if met, 'NO' if not met, or 'PARTIAL' if partially met.
Answer:"""
        
        return self._run_inference(prompt, max_tokens=50)