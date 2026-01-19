#!/bin/bash

# LC Processing API - Startup Script
# This script helps you start the API server

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Header
echo ""
echo "========================================"
echo "   LC Processing API - Startup"
echo "========================================"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    print_info "Docker detected. Would you like to run using Docker? (y/n)"
    read -r use_docker
    
    if [[ $use_docker == "y" || $use_docker == "Y" ]]; then
        print_info "Starting with Docker Compose..."
        
        # Create directories
        mkdir -p uploads results
        
        # Build and start
        docker-compose up --build -d
        
        print_success "API is starting in Docker container..."
        print_info "Waiting for server to be ready..."
        
        # Wait for server to be ready
        sleep 5
        
        # Check if server is running
        if curl -s http://localhost:8000/ > /dev/null 2>&1; then
            print_success "API is running!"
        else
            print_warning "API may still be starting. Check logs with: docker-compose logs -f"
        fi
        
        echo ""
        echo "========================================"
        echo "   Server Running"
        echo "========================================"
        echo ""
        echo "Web Interface:  http://localhost:8000/web_interface.html"
        echo "API Docs:       http://localhost:8000/docs"
        echo "Health Check:   http://localhost:8000/"
        echo ""
        echo "Commands:"
        echo "  View logs:    docker-compose logs -f"
        echo "  Stop server:  docker-compose down"
        echo "  Restart:      docker-compose restart"
        echo ""
        
        exit 0
    fi
fi

# Non-Docker startup
print_info "Starting API server (non-Docker mode)..."
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed!"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
print_info "Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Check and install dependencies
print_info "Checking dependencies..."

if ! pip show fastapi &> /dev/null; then
    print_warning "Dependencies not installed. Installing..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi

# Check system dependencies
print_info "Checking system dependencies..."

MISSING_DEPS=""

if ! command -v tesseract &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS tesseract-ocr"
fi

if ! command -v pdfinfo &> /dev/null; then
    MISSING_DEPS="$MISSING_DEPS poppler-utils"
fi

if [ -n "$MISSING_DEPS" ]; then
    print_warning "Missing system dependencies:$MISSING_DEPS"
    print_info "Install with:"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  sudo apt-get install$MISSING_DEPS"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install${MISSING_DEPS// / brew install }"
    fi
    
    echo ""
    print_info "Continue anyway? (y/n)"
    read -r continue_anyway
    
    if [[ $continue_anyway != "y" && $continue_anyway != "Y" ]]; then
        print_error "Exiting. Please install missing dependencies first."
        exit 1
    fi
else
    print_success "All system dependencies found"
fi

# Create directories
mkdir -p uploads results

# Start the server
echo ""
print_success "Starting LC Processing API..."
echo ""
echo "========================================"
echo "   Server Starting"
echo "========================================"
echo ""
echo "Web Interface:  http://localhost:8000/web_interface.html"
echo "API Docs:       http://localhost:8000/docs"
echo "Health Check:   http://localhost:8000/"
echo ""
echo "Press CTRL+C to stop the server"
echo ""
echo "========================================"
echo ""

# Start the API
python lc_api.py
