#!/bin/bash

echo "========================================"
echo "LC Processing System - Installation"
echo "========================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Linux detected"
    
    # Update package list
    echo "Updating package list..."
    sudo apt-get update
    
    # Install system dependencies
    echo ""
    echo "Installing system dependencies..."
    sudo apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        poppler-utils \
        libgl1-mesa-glx \
        libglib2.0-0
    
    echo "✓ System dependencies installed"
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ macOS detected"
    
    # Install using Homebrew
    if command -v brew &> /dev/null; then
        echo "Installing system dependencies via Homebrew..."
        brew install tesseract poppler
        echo "✓ System dependencies installed"
    else
        echo "⚠ Homebrew not found. Please install Homebrew first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
else
    echo "⚠ Unsupported operating system: $OSTYPE"
    echo "Please manually install:"
    echo "  - Tesseract OCR"
    echo "  - Poppler Utils"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --break-system-packages -r requirements.txt

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Test the installation:"
echo "  python lc_ocr.py --check"
echo ""
echo "Process LC documents:"
echo "  python lc_pipeline.py LC.pdf Amendment.pdf"
echo ""
