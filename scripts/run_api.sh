#!/bin/bash

# Italian Teacher API Startup Script
# Starts the FastAPI server with optional Colab GPU integration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "======================================================"
echo "  Italian Teacher API - Starting Server"
echo "======================================================"
echo ""

# Find Python interpreter
if command -v python3.9 &> /dev/null; then
    PYTHON="python3.9"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    echo -e "${RED}❌ Python not found!${NC}"
    echo "   Please install Python 3.9 or later"
    exit 1
fi

echo -e "${GREEN}✅ Using Python:${NC} $PYTHON ($($PYTHON --version))"
echo ""

# Check if we're in the right directory
if [ ! -f "src/api/main.py" ]; then
    echo -e "${RED}❌ Error: src/api/main.py not found${NC}"
    echo "   Please run this script from the project root directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

# Check for dependencies
echo -e "${BLUE}🔍 Checking dependencies...${NC}"
if ! $PYTHON -c "import fastapi, uvicorn, aiohttp" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Missing dependencies. Installing...${NC}"
    $PYTHON -m pip install -r requirements.txt
    echo ""
fi

# Check for Colab GPU integration
echo "======================================================"
if [ ! -z "$INFERENCE_API_URL" ]; then
    echo -e "${GREEN}🔥 Colab GPU Integration: ENABLED${NC}"
    echo -e "   📡 Inference URL: ${BLUE}$INFERENCE_API_URL${NC}"
    echo ""
    echo "   Testing connection to Colab..."

    # Test if Colab is reachable
    if command -v curl &> /dev/null; then
        if curl -s --max-time 3 "$INFERENCE_API_URL/health" > /dev/null 2>&1; then
            echo -e "   ${GREEN}✅ Colab GPU is reachable!${NC}"
        else
            echo -e "   ${YELLOW}⚠️  Cannot reach Colab GPU${NC}"
            echo "      Make sure your Colab notebook is running"
            echo "      Homework generation will use mock mode"
        fi
    fi
else
    echo -e "${YELLOW}⚠️  Colab GPU Integration: DISABLED${NC}"
    echo "   Homework generation will use mock mode"
    echo ""
    echo -e "   ${BLUE}💡 To enable Colab GPU:${NC}"
    echo "      1. Start your Colab notebook (demos/colab_inference_api.ipynb)"
    echo "      2. Copy the ngrok URL from Colab output"
    echo "      3. Run: export INFERENCE_API_URL=\"https://your-url.ngrok-free.dev\""
    echo "      4. Run: ./run_api.sh"
fi
echo "======================================================"
echo ""

# Start the server
echo -e "${GREEN}🚀 Starting Italian Teacher API...${NC}"
echo ""
echo "   📍 Server URL: http://localhost:8000"
echo "   📚 API Docs:   http://localhost:8000/docs"
echo "   📖 ReDoc:      http://localhost:8000/redoc"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""
echo "======================================================"

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Start uvicorn
# Note: --reload disabled due to Google Drive sync issues
# Restart manually after code changes
$PYTHON -m uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
