#!/bin/bash
# Setup script for healthdq-ai
# Author: Agate Jarmakoviča

set -e  # Exit on error

echo "=================================================="
echo "  healthdq-ai - Setup Script"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

# Check if Python 3.10+
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo -e "${RED}❌ Error: Python 3.10+ required${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python version OK${NC}"
echo

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi
echo

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo -e "${GREEN}✅ Pip upgraded${NC}"
echo

# Install package
echo "📥 Installing healthdq-ai package..."
pip install -e . --quiet
echo -e "${GREEN}✅ Package installed${NC}"
echo

# Install development dependencies
read -p "📦 Install development dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Installing dev dependencies..."
    pip install -e ".[dev]" --quiet
    echo -e "${GREEN}✅ Dev dependencies installed${NC}"
    echo
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created${NC}"
    echo -e "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists${NC}"
fi
echo

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/sample data/ontologies data/feedback
mkdir -p output logs
echo -e "${GREEN}✅ Directories created${NC}"
echo

# Install pre-commit hooks (if dev dependencies installed)
if [ -d "venv/bin" ] && [ -f "venv/bin/pre-commit" ]; then
    echo "🔗 Installing pre-commit hooks..."
    pre-commit install --quiet
    echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
    echo
fi

# Summary
echo "=================================================="
echo "  ✅ Setup Complete!"
echo "=================================================="
echo
echo "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Run tests: ./scripts/run_tests.sh"
echo "  3. Start Streamlit: streamlit run src/healthdq/ui/streamlit_app.py"
echo
echo "For more info, see:"
echo "  - README.md"
echo "  - docs/QUICK_START.md"
echo "  - RUN.md"
echo
echo "Happy data quality improvement! 🎉"
