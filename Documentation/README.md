# Letter of Credit (LC) Processing System

A comprehensive, **fully local** Python system for extracting, processing, and consolidating Letter of Credit (LC) documents and their amendments.

## Features

✨ **Key Capabilities:**
- 📄 **Multi-format Support**: PDF, images, scanned documents, text files
- 🔍 **OCR Engine**: Multiple OCR backends (Tesseract, EasyOCR, PaddleOCR)
- 🏦 **SWIFT Format**: Supports MT700 (LC Issuance) and MT707 (LC Amendment)
- 🔄 **Automatic Consolidation**: Merges original LCs with all amendments
- 📊 **Structured Output**: Clean JSON format for easy integration
- 🎯 **Point-by-Point Tracking**: Tracks changes, additions, and deletions
- 🔒 **100% Local**: No external APIs, all processing happens on your machine

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                Input Documents                       │
│  (PDFs, Images, Scanned Documents, Text Files)      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│           Document Processor (lc_ocr.py)            │
│  • PDF text extraction (pdfplumber)                 │
│  • OCR for scanned documents (Tesseract/EasyOCR)   │
│  • Image preprocessing                               │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│         LC Extractor (lc_extractor.py)              │
│  • Identifies LC vs Amendment                        │
│  • Extracts SWIFT fields (F20, F46A, F47A, etc.)   │
│  • Parses numbered points                            │
│  • Extracts amendment operations (ADD/DELETE/REPALL)│
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│       LC Consolidator (lc_extractor.py)             │
│  • Matches amendments to parent LCs                  │
│  • Applies changes sequentially                      │
│  • Generates final consolidated LC                   │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                  JSON Output                         │
│  • Original LC structure                             │
│  • Amendment details                                 │
│  • Consolidated final LC                             │
│  • Human-readable reports                            │
└─────────────────────────────────────────────────────┘
```

## Installation

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

Or use the automated installer:
```bash
chmod +x install.sh
./install.sh
```

### 3. Verify Installation

```bash
python lc_ocr.py --check
```

## Quick Start

### Basic Usage

```bash
# Process LC documents
python lc_pipeline.py LC_Swift.pdf LC_Amendment_1.pdf

# With custom output directory
python lc_pipeline.py LC.pdf Amendment.pdf --output-dir ./results

# Force OCR (for scanned documents)
python lc_pipeline.py scanned_lc.pdf --force-ocr
```

### Run Demo

```bash
python demo.py
```

This will process the example documents and create:
- `demo_output/original_lc.json` - Original LC structure
- `demo_output/amendment_01.json` - Amendment details
- `demo_output/consolidated_lc.json` - Final consolidated LC

## Usage Examples

### Example 1: Process Single LC with Amendments

```python
from lc_pipeline import LCProcessingPipeline

# Initialize pipeline
pipeline = LCProcessingPipeline(
    ocr_backend='tesseract',
    output_dir='./my_output'
)

# Process documents
files = [
    'LC_0239ILU012702.pdf',
    'Amendment_01.pdf',
    'Amendment_02.pdf'
]

results = pipeline.process_files(files)

print(f"LCs Found: {len(results['lcs_found'])}")
print(f"Consolidated: {len(results['consolidated_lcs'])}")
```

### Example 2: Extract from Text

```python
from lc_extractor import LCExtractor

extractor = LCExtractor()

# From text content
with open('lc_document.txt', 'r') as f:
    text = f.read()

lc_doc = extractor.extract_from_text(text)

print(f"LC Number: {lc_doc.lc_number}")
print(f"Additional Conditions: {len(lc_doc.additional_conditions)}")
```

### Example 3: OCR from Image

```python
from lc_ocr import OCRProcessor

# Initialize OCR
ocr = OCRProcessor(backend='tesseract', language='eng')

# Extract text from image
text = ocr.extract_text_from_image('scanned_lc.jpg')
print(text)
```

## Output Structure

### Consolidated LC JSON

```json
{
  "lc_number": "0239ILU012702",
  "original_issue_date": "2021 Sep 06",
  "sender": "UNILPKKA741",
  "receiver": "EBILSGSGXXX",
  "amendments_applied": 1,
  "last_amendment_date": "2021 Sep 08",
  "fields": {
    "F20": {
      "field_code": "F20",
      "field_name": "Documentary Credit Number",
      "value": "0239ILU012702"
    }
  },
  "additional_conditions": [
    {
      "point_number": 1,
      "text": "PRICE CLAUSE: THE DELIVERED EX-SHIP...",
      "field_code": "F47A"
    },
    {
      "point_number": 19,
      "text": "COMBINED OR SEPARATE DOCUMENTS...",
      "field_code": "F47A",
      "added_by_amendment": true
    }
  ],
  "documents_required": [
    {
      "point_number": 1,
      "text": "SIGNED COMMERCIAL INVOICES...",
      "field_code": "F46A"
    }
  ],
  "amendment_history": [
    {
      "amendment_number": "01",
      "amendment_date": "2021 Sep 08",
      "changes": [
        {
          "operation": "ADD",
          "field_code": "F47B",
          "point_number": 19,
          "narrative": "PLEASE READ FIELD 47A-19 AS..."
        }
      ]
    }
  ]
}
```

## Supported LC Fields

The system extracts all standard SWIFT MT700/MT707 fields:

| Field Code | Field Name | MT700 | MT707 |
|------------|-----------|-------|-------|
| F20 | Documentary Credit Number | ✓ | ✓ |
| F26E | Number of Amendment | - | ✓ |
| F30 | Date of Amendment | - | ✓ |
| F31C | Date of Issue | ✓ | ✓ |
| F31D | Date and Place of Expiry | ✓ | - |
| F32B | Currency Code, Amount | ✓ | - |
| F40A | Form of Documentary Credit | ✓ | - |
| F46A | Documents Required | ✓ | - |
| F46B | Documents Required (Amendment) | - | ✓ |
| F47A | Additional Conditions | ✓ | - |
| F47B | Additional Conditions (Amendment) | - | ✓ |
| F50 | Applicant | ✓ | - |
| F59 | Beneficiary | ✓ | ✓ |

## Amendment Operations

The system recognizes three types of amendment operations:

1. **DELETE**: Remove a specific point
   ```
   /DELETE/ - Removes point N from the original LC
   ```

2. **ADD**: Add a new point
   ```
   /ADD/ - Adds a new point (typically at the end)
   ```

3. **REPALL** (Replace All): Replace entire point content
   ```
   /REPALL/ - Completely replaces point N with new content
   ```

## File Structure

```
lc-processing-system/
├── lc_extractor.py          # Core LC extraction & consolidation
├── lc_ocr.py                # OCR and document processing
├── lc_pipeline.py           # Complete processing pipeline
├── demo.py                  # Demo script
├── requirements.txt         # Python dependencies
├── install.sh              # Installation script
├── README.md               # This file
└── output/                 # Default output directory
    ├── extracted_texts/    # Raw extracted text
    ├── individual_docs/    # Individual LC/Amendment JSONs
    ├── consolidated/       # Consolidated LC JSONs
    └── reports/           # Human-readable reports
```

## OCR Backends

### Tesseract (Default)
- **Pros**: Fast, widely supported, good accuracy
- **Cons**: Requires system installation
- **Best for**: High-quality scans, printed documents

### EasyOCR
- **Pros**: Deep learning-based, high accuracy, supports 80+ languages
- **Cons**: Slower, requires GPU for best performance
- **Best for**: Complex layouts, poor quality scans

### PaddleOCR
- **Pros**: Very fast, excellent for Chinese/English mix
- **Cons**: Larger model size
- **Best for**: Asian language documents, high-volume processing

### Usage
```bash
# Use Tesseract (default)
python lc_pipeline.py document.pdf

# Use EasyOCR
python lc_pipeline.py document.pdf --ocr-backend easyocr

# Use PaddleOCR
python lc_pipeline.py document.pdf --ocr-backend paddleocr
```

## Advanced Features

### 1. Batch Processing

```python
from lc_pipeline import LCProcessingPipeline
from pathlib import Path

pipeline = LCProcessingPipeline()

# Get all PDFs in a directory
pdf_files = list(Path('./lc_documents').glob('*.pdf'))

# Process all at once
results = pipeline.process_files([str(f) for f in pdf_files])
```

### 2. Custom Output Formats

```python
# Generate human-readable report
pipeline.generate_human_readable_report(lc_number='0239ILU012702')

# Export specific fields
consolidated = pipeline.consolidator.consolidate('0239ILU012702')
conditions = consolidated['additional_conditions']
```

### 3. Error Handling

```python
results = pipeline.process_files(files)

# Check for errors
if results['errors']:
    print("Errors encountered:")
    for error in results['errors']:
        print(f"  - {error}")
```

## Troubleshooting

### Issue: "Tesseract not found"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Issue: "pdf2image not working"
**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

### Issue: Poor OCR accuracy
**Solutions:**
1. Use higher DPI (edit `lc_ocr.py`, change `dpi=300` to `dpi=600`)
2. Try different OCR backend: `--ocr-backend easyocr`
3. Preprocess images before OCR
4. Check document quality and rescan if possible

### Issue: LC number not detected
**Solutions:**
1. Check if document follows SWIFT MT700/MT707 format
2. Look for F20 field in the text
3. Manually verify the extracted text in `extracted_texts/` folder

## Performance

Typical processing times (on standard hardware):

| Document Type | Pages | Processing Time |
|--------------|-------|-----------------|
| Digital PDF | 4 | 1-2 seconds |
| Scanned PDF | 4 | 15-30 seconds |
| High-res Image | 1 | 5-10 seconds |

## Limitations

1. **Format Dependency**: Works best with SWIFT MT700/MT707 formats
2. **OCR Accuracy**: Dependent on document quality
3. **Language**: Optimized for English text
4. **Structured Data**: Requires consistent numbering in conditions/documents

## Future Enhancements

- [ ] Web UI for drag-and-drop processing
- [ ] Support for additional LC formats (UCP 600, eUCP)
- [ ] Machine learning for field detection
- [ ] Export to Excel/CSV
- [ ] Database integration
- [ ] Multi-language OCR support
- [ ] Comparison tool for LC versions

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - feel free to use in your projects

## Support

For issues, questions, or feature requests, please create an issue in the repository.

## Example Output

After running the demo:

```
================================================================================
PROCESSING COMPLETE
================================================================================

Summary:
  Total Files: 2
  LCs Found: 1
  Amendments Found: 1
  Consolidated LCs: 1
  Errors: 0

Output Directory: ./demo_output
  ✓ original_lc.json
  ✓ amendment_01.json
  ✓ consolidated_lc.json
  ✓ 0239ILU012702_report.txt
```

## Credits

Built with:
- Python 3.8+
- Tesseract OCR
- pdfplumber
- pdf2image
- Pillow
- NumPy

---

**Note**: This is a local processing system. No data leaves your machine. Perfect for sensitive financial documents.
