"""
LC Consolidator
Consolidates LC with amendments using multi-dialect semantic patching and AI Auditing
"""

import re
from typing import Dict, Any, List
from dataclasses import asdict
from .utils import clean_instruction_text, extract_point_number
from services.ai_auditor import OfflineLCAuditor


class LCConsolidator:
    """
    Consolidate LC with amendments using multi-dialect semantic patching and AI Auditing.
    Handles squashed text, complex replacements, and multiple operational scenarios.
    """
    
    def __init__(self, use_ai=True):
        self.lcs: Dict[str, Any] = {}
        self.amendments: Dict[str, List[Any]] = {}
        self.use_ai = use_ai
        if self.use_ai:
            self.auditor = OfflineLCAuditor()
    
    def add_document(self, doc: Any):
        """Categorizes documents into LCs or Amendments for consolidation."""
        ln = doc.lc_number.strip() if hasattr(doc, 'lc_number') else str(doc)
        if doc.document_type == "LC":
            self.lcs[ln] = doc
        else:
            if ln not in self.amendments:
                self.amendments[ln] = []
            self.amendments[ln].append(doc)

    def _apply_change(self, points_list: List[Dict], change: Dict, field_code: str):
        """
        Intelligently applies an amendment change to the existing points list.
        Prevents 'half-text' errors by ensuring the original 200-word clause is 
        preserved unless a total replacement is explicitly required.
        """
        narrative = change.get('narrative', '')
        # Clean up common OCR artifacts and quote doubling
        raw_text = change.get('change_text', narrative).replace("''", "'").replace('"', "'")
        
        # 1. IDENTIFY TARGET POINT
        # Use the extractor's point number or parse it from the text (e.g., "CLAUSE 10")
        target_point = change.get('point_number')
        if not target_point:
            target_point = extract_point_number(raw_text)
        
        # 2. FIND THE EXISTING CLAUSE IN THE LC
        existing_point = None
        if target_point:
            existing_point = next((p for p in points_list if p.get('point_number') == target_point), None)

        # 3. OPERATION: DELETE
        operation = change.get('operation', '').upper()
        if operation == 'DELETE' and target_point:
            # Use a list comprehension to remove the point without index errors
            points_list[:] = [p for p in points_list if p.get('point_number') != target_point]
            print(f"✓ Deleted point {target_point}")
            return

        # 4. OPERATION: MODIFY EXISTING POINT (The "Half-Text" Fix)
        if existing_point:
            original_text = existing_point.get('text', '')
            
            # Scenario A: AI Semantic Merging (Best for name changes inside long clauses)
            if self.use_ai:
                try:
                    merged_text = self.auditor.generate_merged_text(
                        original_text=original_text,
                        instruction=raw_text
                    )
                    
                    if merged_text and len(merged_text) > (len(raw_text) * 0.5):
                        existing_point.update({
                            'text': merged_text,
                            'modified_by_amendment': True,
                            'ai_processed': True,
                            'original_instruction': raw_text
                        })
                        print(f"✓ AI semantic merge successful for point {target_point}")
                        return
                except Exception as e:
                    print(f"⚠ AI merge failed for point {target_point}: {e}")

            # Scenario B: Regex Fallback (Handles "X INSTEAD OF Y")
            result = self._regex_fallback(original_text, raw_text)
            if result:
                existing_point.update({
                    'text': result,
                    'modified_by_amendment': True,
                    'fallback_method': 'regex'
                })
                print(f"✓ Regex patched point {target_point}")
                return

            # Scenario C: Total Replacement (If it says "TO READ AS" but merging failed)
            if "TO READ AS" in raw_text.upper() or operation == 'REPALL':
                clean_text = clean_instruction_text(raw_text)
                # Validation: Don't replace a long text with a tiny "instruction-like" string
                if len(clean_text) > 5:
                    existing_point['text'] = clean_text
                    existing_point['modified_by_amendment'] = True
                    print(f"✓ Point {target_point} replaced with new text block")
                return

        # 5. OPERATION: ADD NEW POINT
        else:
            new_text = clean_instruction_text(raw_text)
            # Ensure we don't add "empty" instructions or pure "DELETE" markers
            if new_text and "/DELETE/" not in raw_text.upper():
                new_point = {
                    'point_number': target_point or (max([p.get('point_number', 0) for p in points_list], default=0) + 1),
                    'text': new_text,
                    'field_code': field_code,
                    'added_by_amendment': True,
                    'original_instruction': raw_text
                }
                points_list.append(new_point)
                print(f"✓ Added new point {new_point['point_number']}")
   
    def _regex_fallback(self, original_text: str, instruction: str) -> str:
        """
        Attempts regex-based replacement when AI fails.
        """
        # Pattern 1: "X INSTEAD OF Y"
        match = re.search(r"['\"](.+?)['\"]\s+INSTEAD\s+OF\s+['\"](.+?)['\"]", instruction, re.I | re.S)
        if match:
            new_val, old_val = match.group(1).strip(), match.group(2).strip()
            if old_val.upper() in original_text.upper():
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
        # Pattern 2: "DELETE X REPLACE BY Y"
        match = re.search(r"DELETE\s+['\"](.+?)['\"]\s+REPLACE\s+(?:BY|WITH)\s+['\"](.+?)['\"]", instruction, re.I | re.S)
        if match:
            old_val, new_val = match.group(1).strip(), match.group(2).strip()
            if old_val.upper() in original_text.upper():
                return re.sub(re.escape(old_val), new_val, original_text, flags=re.IGNORECASE)
        
        # Pattern 3: "TO READ AS X" (complete replacement)
        match = re.search(r"TO\s+READ\s+AS\s+['\"](.+?)['\"]", instruction, re.I | re.S)
        if match:
            return match.group(1).strip()
        
        return None

    def consolidate(self, lc_number: str) -> Dict:
        """Merges all amendments into the base LC and returns a consolidated document."""
        if lc_number not in self.lcs:
            return None
            
        print(f"\n{'='*80}")
        print(f"CONSOLIDATING LC: {lc_number}")
        print(f"{'='*80}\n")
        
        original_lc = self.lcs[lc_number]
        amendments = self.amendments.get(lc_number, [])
        
        # Sort amendments by number to ensure chronological patching
        amendments.sort(key=lambda x: int(x.amendment_number) if x.amendment_number else 0)
        
        consolidated = {
            'lc_number': lc_number,
            'original_issue_date': original_lc.issue_date,
            'sender': original_lc.sender,
            'receiver': original_lc.receiver,
            'message_type': 'MT700_CONSOLIDATED',
            'amendments_applied': len(amendments),
            'last_amendment_date': amendments[-1].amendment_date if amendments else None,
            'fields': {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v 
                      for k, v in original_lc.fields.items()},
            'additional_conditions': [dict(p) for p in original_lc.additional_conditions],
            'documents_required': [dict(p) for p in original_lc.documents_required],
            'amendment_history': []
        }
        
        for amendment in amendments:
            print(f"\n--- Processing Amendment {amendment.amendment_number} ---")
            am_rec = {'amendment_number': amendment.amendment_number, 'changes': []}
            
            # Update Field 47A (Additional Conditions) using Amendment Field 47B
            for change in amendment.additional_conditions:
                print(f"  Applying change to 47A: {change.get('narrative', '')[:80]}...")
                self._apply_change(consolidated['additional_conditions'], change, '47A')
                am_rec['changes'].append(change)
            
            # Update Field 46A (Documents Required) using Amendment Field 46B
            for change in amendment.documents_required:
                print(f"  Applying change to 46A: {change.get('narrative', '')[:80]}...")
                self._apply_change(consolidated['documents_required'], change, '46A')
                am_rec['changes'].append(change)
            
            consolidated['amendment_history'].append(am_rec)
        
        # Final cleanup: Ensure points are in numerical order
        consolidated['additional_conditions'].sort(key=lambda x: x.get('point_number', 999))
        consolidated['documents_required'].sort(key=lambda x: x.get('point_number', 999))
        
        print(f"\n{'='*80}")
        print(f"CONSOLIDATION COMPLETE")
        print(f"{'='*80}\n")
        
        return consolidated

    def get_all_consolidated(self) -> List[Dict]:
        """Iterates through all loaded LCs and returns their consolidated state."""
        return [self.consolidate(ln) for ln in self.lcs.keys() if self.consolidate(ln)]

    def print_summary(self, consolidated: Dict):
        """Prints a human-readable summary of the consolidated LC."""
        print(f"\n{'='*80}")
        print(f"CONSOLIDATED LC SUMMARY: {consolidated['lc_number']}")
        print(f"{'='*80}")
        print(f"Amendments Applied: {consolidated['amendments_applied']}")
        print(f"Last Amendment: {consolidated['last_amendment_date']}")
        
        print(f"\n--- Additional Conditions (Field 47A) ---")
        for condition in consolidated['additional_conditions']:
            status = ""
            if condition.get('modified_by_amendment'):
                status = "[MODIFIED]"
            elif condition.get('added_by_amendment'):
                status = "[NEW]"
            
            print(f"{condition['point_number']:3d}. {status:12s} {condition['text'][:100]}...")
        
        print(f"\n--- Documents Required (Field 46A) ---")
        for doc in consolidated['documents_required']:
            status = ""
            if doc.get('modified_by_amendment'):
                status = "[MODIFIED]"
            elif doc.get('added_by_amendment'):
                status = "[NEW]"
            
            print(f"{doc['point_number']:3d}. {status:12s} {doc['text'][:100]}...")
        
        print(f"\n{'='*80}\n")