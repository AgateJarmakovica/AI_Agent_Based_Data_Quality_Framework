#!/bin/bash
# Streamlit launcher script for healthdq-ai
# Author: Agate Jarmakoviča

echo "=================================================="
echo "  healthdq-ai - Streamlit Launcher"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}❌ Streamlit not installed${NC}"
    echo "Installing streamlit..."
    pip install streamlit pandas pyyaml
    echo -e "${GREEN}✅ Streamlit installed${NC}"
    echo
fi

# Default values
PORT=8501
HOST="localhost"
FILE="src/healthdq/ui/streamlit_app.py"
MULTIPAGE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --multipage|-m)
            MULTIPAGE=true
            FILE="src/healthdq/ui/pages/1_📤_Upload.py"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --port, -p PORT      Port number (default: 8501)"
            echo "  --host HOST          Host address (default: localhost)"
            echo "  --multipage, -m      Run multipage version"
            echo "  --help, -h           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if file exists
if [ ! -f "$FILE" ]; then
    echo -e "${RED}❌ File not found: $FILE${NC}"
    exit 1
fi

# Display info
echo "🚀 Starting Streamlit..."
echo "   File: $FILE"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo

if $MULTIPAGE; then
    echo -e "${YELLOW}📄 Running multipage version${NC}"
    echo "   Pages: src/healthdq/ui/pages/"
else
    echo -e "${GREEN}📄 Running single-page version${NC}"
fi

echo
echo "=================================================="
echo "  🌐 Access the app at:"
echo "     http://${HOST}:${PORT}"
echo "=================================================="
echo
echo "Press Ctrl+C to stop"
echo

# Run streamlit
streamlit run "$FILE" \
    --server.port "$PORT" \
    --server.address "$HOST" \
    --browser.gatherUsageStats false
