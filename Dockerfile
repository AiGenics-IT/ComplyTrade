FROM python:3.9-slim

# Set project root as working directory
WORKDIR /app

# -----------------------
# System dependencies
# -----------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# -----------------------
# Optional Python dependencies
# -----------------------
COPY requirements/requirements.txt ./requirements.txt
# RUN pip install --no-cache-dir --upgrade pip && \
    # pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------
# Copy the full project
# -----------------------
COPY . .

# -----------------------
# Runtime directories
# -----------------------
RUN mkdir -p /tmp/lc_uploads /tmp/lc_results

# -----------------------
# PYTHONPATH for multi-folder imports
# -----------------------
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# -----------------------
# Start the API
# -----------------------
CMD ["python", "RestApiSystem/lc_api.py"]
