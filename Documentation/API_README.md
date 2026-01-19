# LC Processing API - Quick Start Guide

🚀 **REST API for processing Letter of Credit documents with a beautiful web interface**

## What You Get

✅ **REST API** - Upload files via HTTP requests  
✅ **Web Interface** - Drag & drop interface for easy uploads  
✅ **Background Processing** - Non-blocking document processing  
✅ **JSON Output** - Structured data for easy integration  
✅ **Python Client** - Pre-built client library  
✅ **Docker Support** - One-command deployment  

---

## 🎯 Quick Start (3 Steps)

### Option 1: Using Docker (Easiest)

```bash
# 1. Start the API
docker-compose up -d

# 2. Open web interface
open http://localhost:8000/web_interface.html

# Done! 🎉
```

### Option 2: Using Python

```bash
# 1. Start the API
chmod +x start_api.sh
./start_api.sh

# 2. Open web interface
open http://localhost:8000/web_interface.html

# Done! 🎉
```

---

## 📁 What's Included

```
.
├── lc_api.py              # API server (FastAPI)
├── lc_extractor.py        # LC extraction engine
├── lc_ocr.py              # OCR processing
├── lc_api_client.py       # Python client library
├── web_interface.html     # Web UI
├── start_api.sh           # Startup script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
├── requirements.txt       # Python dependencies
└── API_DOCUMENTATION.md   # Full API docs
```

---

## 🌐 Using the Web Interface

1. **Start the server** (see Quick Start above)

2. **Open your browser** to `http://localhost:8000/web_interface.html`

3. **Upload files**:
   - Drag & drop files into the upload area
   - Or click to select files
   - Supported: PDF, JPG, PNG, TIFF

4. **Configure options**:
   - Choose OCR engine (Tesseract, EasyOCR, PaddleOCR)
   - Enable "Force OCR" for scanned documents

5. **Process**:
   - Click "Process Documents"
   - Watch real-time progress
   - Download results when complete

---

## 🐍 Using Python Client

### Basic Usage

```python
from lc_api_client import LCProcessingClient

# Initialize client
client = LCProcessingClient(base_url="http://localhost:8000")

# Upload and process files
results = client.process_and_wait([
    "LC_Swift.pdf",
    "LC_Amendment_1.pdf"
])

# Print results
print(f"LCs found: {results['lcs_found']}")
print(f"Amendments: {results['amendments_found']}")

# Download consolidated LC
for lc in results['consolidated_lcs']:
    client.download_lc(
        results['job_id'],
        lc['lc_number'],
        f"{lc['lc_number']}.json"
    )
```

### Command Line

```bash
# Process files
python lc_api_client.py LC.pdf Amendment.pdf

# With options
python lc_api_client.py *.pdf --ocr-backend easyocr --output-dir results
```

---

## 🔌 Using the REST API

### Upload Documents

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "files=@LC_Swift.pdf" \
  -F "files=@Amendment.pdf"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Processing 2 files"
}
```

### Check Status

```bash
curl "http://localhost:8000/api/status/550e8400-e29b-41d4-a716-446655440000"
```

### Get Results

```bash
curl "http://localhost:8000/api/result/550e8400-e29b-41d4-a716-446655440000"
```

### Download Consolidated LC

```bash
curl -O "http://localhost:8000/api/download/JOB_ID/LC_NUMBER"
```

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/api/upload` | Upload documents |
| GET | `/api/status/{job_id}` | Check job status |
| GET | `/api/result/{job_id}` | Get results |
| GET | `/api/download/{job_id}/{lc_number}` | Download LC JSON |
| GET | `/api/lc/{job_id}/{lc_number}` | Get LC data |
| GET | `/api/jobs` | List all jobs |
| DELETE | `/api/job/{job_id}` | Delete job |

**Full documentation:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)

---

## 🎨 Web Interface Features

- ✅ **Drag & Drop** file upload
- ✅ **Real-time progress** tracking
- ✅ **Multiple file** support
- ✅ **OCR engine** selection
- ✅ **Results preview** with statistics
- ✅ **One-click download** of consolidated LCs
- ✅ **Beautiful, modern** UI

---

## 🐳 Docker Commands

### Start Server
```bash
docker-compose up -d
```

### View Logs
```bash
docker-compose logs -f
```

### Stop Server
```bash
docker-compose down
```

### Restart Server
```bash
docker-compose restart
```

### Rebuild
```bash
docker-compose up --build -d
```

---

## 🔧 Installation (Manual)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

### Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Start Server

```bash
python lc_api.py
```

---

## 🎯 Usage Examples

### Example 1: Web Interface (Easiest)

1. Open `http://localhost:8000/web_interface.html`
2. Drag your LC and amendment files
3. Click "Process Documents"
4. Download results

### Example 2: Command Line

```bash
python lc_api_client.py \
  LC_0239ILU012702.pdf \
  Amendment_01.pdf \
  --output-dir ./results
```

### Example 3: Python Script

```python
from lc_api_client import LCProcessingClient

client = LCProcessingClient()

# Upload files
job_info = client.upload_documents([
    "LC.pdf",
    "Amendment.pdf"
])

# Wait and get results
import time
while True:
    status = client.get_status(job_info['job_id'])
    if status['status'] == 'completed':
        break
    time.sleep(2)

results = client.get_result(job_info['job_id'])
print(results)
```

### Example 4: cURL

```bash
# Upload
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/upload" \
  -F "files=@LC.pdf" \
  -F "files=@Amendment.pdf")

# Extract job ID
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# Check status
curl "http://localhost:8000/api/status/$JOB_ID"

# Get results
curl "http://localhost:8000/api/result/$JOB_ID" | jq
```

---

## 🔍 Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide:
- ✅ Complete API reference
- ✅ Try-it-out functionality
- ✅ Request/response examples
- ✅ Schema documentation

---

## 📊 Response Structure

### Consolidated LC

```json
{
  "lc_number": "0239ILU012702",
  "original_issue_date": "2021 Sep 06",
  "amendments_applied": 1,
  
  "additional_conditions": [
    {
      "point_number": 1,
      "text": "PRICE CLAUSE: ...",
      "field_code": "F47A"
    },
    {
      "point_number": 19,
      "text": "New condition",
      "field_code": "F47A",
      "added_by_amendment": true
    }
  ],
  
  "amendment_history": [
    {
      "amendment_number": "01",
      "amendment_date": "2021 Sep 08",
      "changes": [...]
    }
  ]
}
```

---

## ⚡ Performance

| Document Type | Files | Processing Time |
|--------------|-------|-----------------|
| Digital PDF | 2 | 2-5 seconds |
| Scanned PDF | 2 | 30-60 seconds |
| Mixed (10 files) | 10 | 1-2 minutes |

---

## 🛠️ Configuration

### Change Port

Edit `lc_api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change to 8080
```

### Change Storage Location

Edit `lc_api.py`:
```python
UPLOAD_DIR = Path("/your/custom/path/uploads")
RESULTS_DIR = Path("/your/custom/path/results")
```

### OCR Backend

Choose when uploading:
- **Tesseract** (default): Fast, good for printed docs
- **EasyOCR**: High accuracy, slower
- **PaddleOCR**: Best for multi-language

---

## 🐛 Troubleshooting

### Server Won't Start

**Error**: `Address already in use`

**Solution**:
```bash
# Find process on port 8000
lsof -i :8000

# Kill it or use different port
```

### OCR Not Working

**Error**: `Tesseract not found`

**Solution**:
```bash
# Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Slow Processing

**Issue**: Taking too long

**Solutions**:
- Use `force_ocr=false` for digital PDFs
- Use Tesseract instead of EasyOCR
- Process fewer files at once
- Upgrade server resources

---

## 🔒 Security Notes

### For Production Use:

1. **Add Authentication**:
   ```python
   from fastapi.security import HTTPBearer
   ```

2. **Rate Limiting**:
   ```bash
   pip install slowapi
   ```

3. **HTTPS**: Use reverse proxy (nginx)

4. **File Size Limits**: Configure in FastAPI

5. **Input Validation**: Already included

---

## 📦 Deployment

### Production Server

```bash
# Install Gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn lc_api:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📚 Additional Resources

- **Full API Documentation**: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Python Client Examples**: See `lc_api_client.py`
- **Core System README**: [README.md](README.md)

---

## ✨ Key Features

- ✅ **RESTful API** with FastAPI
- ✅ **Background processing** with job queues
- ✅ **Multiple file formats** (PDF, images)
- ✅ **OCR support** (3 engines)
- ✅ **SWIFT MT700/MT707** format support
- ✅ **Automatic consolidation** of amendments
- ✅ **JSON output** for easy integration
- ✅ **Web interface** for non-developers
- ✅ **Docker support** for easy deployment
- ✅ **Python client** library included

---

## 🎉 You're Ready!

1. Start the server: `./start_api.sh`
2. Open web interface: `http://localhost:8000/web_interface.html`
3. Upload your LC documents
4. Get structured JSON results

**Questions?** Check the [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed information.

---

**Version**: 1.0.0  
**License**: MIT
