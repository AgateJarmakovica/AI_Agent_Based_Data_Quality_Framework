#!/bin/bash
# Quick Start Script for healthdq-ai Streamlit UI
# Author: Agate Jarmakoviča

set -e

echo "=============================================="
echo "  healthdq-ai Streamlit Quick Start"
echo "=============================================="
echo ""

# Check if in correct directory
if [ ! -f "src/healthdq/ui/streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found!"
    echo "   Please run this script from the project root directory:"
    echo "   cd /home/user/AI_Agent_Based_Data_Quality_Framework"
    exit 1
fi

echo "📁 Project directory: OK"
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"
echo ""

# Check if Streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not installed"
    echo "📦 Installing minimal dependencies..."
    echo ""

    pip install streamlit pandas pyyaml numpy -q

    echo "✅ Minimal dependencies installed"
    echo ""
else
    STREAMLIT_VERSION=$(python -c "import streamlit; print(streamlit.__version__)")
    echo "✅ Streamlit already installed: v$STREAMLIT_VERSION"
    echo ""
fi

# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "🚀 Starting Streamlit application..."
echo ""
echo "   Local URL: http://localhost:8501"
echo "   To stop: Press Ctrl+C"
echo ""
echo "=============================================="
echo ""

# Launch Streamlit
streamlit run src/healthdq/ui/streamlit_app.py
