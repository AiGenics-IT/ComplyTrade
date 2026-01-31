"""
Utility Functions
Helper functions for text processing and normalization
"""

import re


def normalize_lc_number(value: str) -> str:
    """Normalize LC number by removing special characters"""
    if not value:
        return ""
    return re.sub(r'[^A-Z0-9]', '', value.upper())


def sanitize_text(text: str) -> str:
    """Removes non-mappable characters that break the console."""
    if not text:
        return ""
    # Only keep printable characters, tabs, and newlines
    return "".join(c for c in text if c.isprintable() or c in "\n\r\t")


def strip_ocr_labels(text: str) -> str:
    """Remove common OCR labels from text"""
    # List of common labels found in OCR
    labels = [
        r'Documentary Credit Number', r"Sender's Reference", 
        r'Additional Conditions', r'Documents Required',
        r'Description of Goods', r'Date of Issue'
    ]
    clean_text = text
    for label in labels:
        # Case-insensitive removal of the label if it's at the start
        clean_text = re.sub(rf'^\s*{label}\s*', '', clean_text, flags=re.IGNORECASE)
    return clean_text.strip()


def clean_field_value(value: str) -> str:
    """Enhanced cleaning for MT700 and Supporting Docs to prevent label-bleeding."""
    if not value:
        return ""

    # 1. Remove OCR Label noise globally
    noise_patterns = [
        r'Lines\s?\d?\s?to\s?\d+:?', 
        r'Lines\s?\d?-\d+:?',
        r'Lines\d+[a-z]*',   # Catches squashed 'Lines2t'
        r'Narrativel?:?', 
        r'Code\s?:?'
    ]
    
    for pattern in noise_patterns:
        value = re.sub(pattern, '', value, flags=re.IGNORECASE)

    # 2. Normalize whitespace
    value = re.sub(r'\s+', ' ', value)

    # 3. Strip common labels that appear at the start of values
    prefixes = [
        'Name and Address:', 'Currency:', 'Date:', 'Place:', 
        'Number:', 'Total:', 'Amount:', 'Days:', 'Party Identifier:',
        'Account:', 'Settlement Amount:', 'Narrative:', 'Beneficiary:'
    ]
    
    for prefix in prefixes:
        value = re.sub(f'^{re.escape(prefix)}\s*', '', value, flags=re.IGNORECASE)

    # 4. Final Polish
    value = value.strip()
    # Remove any leading artifacts common in PDF-to-Text conversion
    value = re.sub(r'^[+)\s/:\-]+', '', value) 
    
    return value.strip()


def clean_instruction_text(text: str) -> str:
    """Removes 'CLAUSE NO. X TO READ AS' prefixes to isolate the pure value."""
    text = re.sub(r'^(?:CLAUSE|ITEM)\s*(?:NO\.)?\s*\d+\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?:NOW\s+)?TO\s+READ\s+AS\s*', '', text, flags=re.IGNORECASE)
    return text.strip()


def extract_point_number(instruction: str) -> int:
    """
    Extracts the clause/point number from the instruction.
    Example: "CLAUSE NO.5" -> 5, "CLAUSENO.27" -> 27
    """
    match = re.search(r'(?:CLAUSE|ITEM)?\s*(?:NO\.?)?\s*(\d+)', instruction, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None