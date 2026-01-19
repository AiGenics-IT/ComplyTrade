# Quick Start Guide - LC Processing System

## What This System Does

This system extracts and consolidates Letter of Credit (LC) documents:
1. **Extracts** all points from original LCs
2. **Identifies** amendments and their changes
3. **Consolidates** everything into a final LC with all amendments applied
4. **Outputs** structured JSON files

## Installation (5 minutes)

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

### Step 2: Install Python Packages

```bash
pip install pdfplumber PyPDF2 pdf2image pytesseract Pillow numpy
```

**OR** use the automated installer:
```bash
chmod +x install.sh
./install.sh
```

## Basic Usage

### Option 1: Use the Demo (Quickest)

```bash
python demo.py
```

This processes the example LC and amendment, creating:
- `demo_output/original_lc.json`
- `demo_output/amendment_01.json`
- `demo_output/consolidated_lc.json`

### Option 2: Process Your Own Documents

```bash
python lc_pipeline.py your_lc.pdf your_amendment.pdf
```

Output will be in `./lc_output/` directory:
- `extracted_texts/` - Raw text from documents
- `individual_docs/` - Each document as JSON
- `consolidated/` - Final consolidated LCs
- `reports/` - Human-readable reports

## Understanding the Output

### Consolidated LC JSON Structure:

```json
{
  "lc_number": "0239ILU012702",
  "original_issue_date": "2021 Sep 06",
  "amendments_applied": 1,
  
  "additional_conditions": [
    {
      "point_number": 1,
      "text": "Original condition text...",
      "field_code": "F47A"
    },
    {
      "point_number": 19,
      "text": "New condition added by amendment...",
      "field_code": "F47A",
      "added_by_amendment": true  ← Added by amendment
    }
  ],
  
  "amendment_history": [
    {
      "amendment_number": "01",
      "changes": [
        {
          "operation": "ADD",
          "point_number": 19,
          "narrative": "Details of the change..."
        }
      ]
    }
  ]
}
```

## Common Tasks

### Task 1: Process Multiple Amendments

```bash
python lc_pipeline.py LC.pdf Amendment1.pdf Amendment2.pdf Amendment3.pdf
```

The system automatically:
- Identifies which is the original LC
- Matches amendments to the correct LC
- Applies changes in order

### Task 2: Handle Scanned/Image PDFs

```bash
python lc_pipeline.py scanned_lc.pdf --force-ocr
```

### Task 3: Use Different Output Directory

```bash
python lc_pipeline.py LC.pdf Amendment.pdf --output-dir ./my_results
```

### Task 4: Check if OCR is Working

```bash
python lc_ocr.py --check
```

## What Gets Extracted?

### From Original LC (MT700):
- LC Number (F20)
- Issue Date (F31C)
- Expiry Date (F31D)
- Amount (F32B)
- Applicant (F50)
- Beneficiary (F59)
- **Additional Conditions (F47A)** - All numbered points
- **Documents Required (F46A)** - All numbered points
- All other SWIFT fields

### From Amendments (MT707):
- Amendment Number (F26E)
- Amendment Date (F30)
- Changes to Additional Conditions (F47B)
  - DELETE operations
  - ADD operations
  - REPLACE operations
- Changes to Documents Required (F46B)

### In the Consolidated Output:
- Original LC with all amendments applied
- Track which points were added/modified
- Complete amendment history
- All SWIFT fields preserved

## Troubleshooting

### Problem: "No text extracted"
**Solution**: Your PDF might be scanned. Use `--force-ocr`:
```bash
python lc_pipeline.py document.pdf --force-ocr
```

### Problem: "Tesseract not found"
**Solution**: Install Tesseract:
```bash
# Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Problem: "LC number not detected"
**Solution**: 
1. Check if your document is in SWIFT MT700/MT707 format
2. Look for "F20:" field with the LC number
3. View extracted text in `lc_output/extracted_texts/` to verify

### Problem: Points not extracted correctly
**Solution**:
1. Ensure points are numbered (1., 2., 3., etc.)
2. Check the raw text extraction
3. May need to adjust regex patterns for your specific format

## Example Workflow

1. **Collect Documents**: Gather all LC and amendment PDFs
   ```
   documents/
   ├── LC_0239ILU012702.pdf
   ├── Amendment_01.pdf
   └── Amendment_02.pdf
   ```

2. **Process**: Run the pipeline
   ```bash
   python lc_pipeline.py documents/*.pdf --output-dir results
   ```

3. **Review**: Check the outputs
   ```bash
   ls results/consolidated/
   # Shows: 0239ILU012702_consolidated.json
   ```

4. **Integrate**: Use the JSON in your application
   ```python
   import json
   
   with open('results/consolidated/0239ILU012702_consolidated.json') as f:
       lc_data = json.load(f)
   
   # Access the data
   print(f"LC: {lc_data['lc_number']}")
   print(f"Conditions: {len(lc_data['additional_conditions'])}")
   ```

## Key Features

✅ **Fully Local** - No cloud services, all processing on your machine  
✅ **Multiple Formats** - PDF, images, scanned documents  
✅ **OCR Support** - Handles both digital and scanned documents  
✅ **Automatic Matching** - Links amendments to correct LCs  
✅ **Point Tracking** - Knows which points were added/modified  
✅ **JSON Output** - Easy to integrate with other systems  
✅ **Human-Readable Reports** - Also generates text reports  

## Next Steps

1. Run the demo to see how it works
2. Try with your own LC documents
3. Integrate the JSON output into your workflow
4. Customize the extraction patterns if needed

## Need Help?

- Check the full README.md for detailed documentation
- Review the demo.py code to see usage examples
- Look at the generated JSON files to understand the structure

## File Reference

- `lc_pipeline.py` - Main entry point (use this)
- `lc_extractor.py` - Core extraction logic
- `lc_ocr.py` - OCR and document processing
- `demo.py` - Working example
- `requirements.txt` - Python dependencies
- `install.sh` - Automated installation

---

**Remember**: Start with `python demo.py` to see the system in action!
