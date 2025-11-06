#!/bin/bash
# Cleanup script for healthdq-ai
# Author: Agate Jarmakoviča

echo "=================================================="
echo "  healthdq-ai - Cleanup Script"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
CLEAN_ALL=false
CLEAN_CACHE=false
CLEAN_BUILD=false
CLEAN_LOGS=false
CLEAN_DATA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            CLEAN_ALL=true
            shift
            ;;
        --cache)
            CLEAN_CACHE=true
            shift
            ;;
        --build)
            CLEAN_BUILD=true
            shift
            ;;
        --logs)
            CLEAN_LOGS=true
            shift
            ;;
        --data)
            CLEAN_DATA=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --all      Clean everything"
            echo "  --cache    Clean Python cache (__pycache__, *.pyc)"
            echo "  --build    Clean build artifacts (dist/, *.egg-info)"
            echo "  --logs     Clean log files"
            echo "  --data     Clean generated data (careful!)"
            echo "  --help, -h Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to cache if nothing specified
if ! $CLEAN_ALL && ! $CLEAN_CACHE && ! $CLEAN_BUILD && ! $CLEAN_LOGS && ! $CLEAN_DATA; then
    CLEAN_CACHE=true
fi

# Clean Python cache
if $CLEAN_ALL || $CLEAN_CACHE; then
    echo "🗑️  Cleaning Python cache..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    rm -rf .pytest_cache .mypy_cache .coverage htmlcov/ 2>/dev/null || true
    echo -e "${GREEN}✅ Python cache cleaned${NC}"
fi

# Clean build artifacts
if $CLEAN_ALL || $CLEAN_BUILD; then
    echo "🗑️  Cleaning build artifacts..."
    rm -rf build/ dist/ *.egg-info 2>/dev/null || true
    rm -rf .eggs/ 2>/dev/null || true
    echo -e "${GREEN}✅ Build artifacts cleaned${NC}"
fi

# Clean logs
if $CLEAN_ALL || $CLEAN_LOGS; then
    echo "🗑️  Cleaning log files..."
    rm -rf logs/*.log 2>/dev/null || true
    rm -rf *.log 2>/dev/null || true
    echo -e "${GREEN}✅ Log files cleaned${NC}"
fi

# Clean data (with confirmation)
if $CLEAN_ALL || $CLEAN_DATA; then
    echo -e "${YELLOW}⚠️  WARNING: This will delete generated data!${NC}"
    read -p "Are you sure? (type 'yes' to confirm): " -r
    echo
    if [[ $REPLY == "yes" ]]; then
        echo "🗑️  Cleaning generated data..."
        rm -rf data/feedback/*.json 2>/dev/null || true
        rm -rf output/* 2>/dev/null || true
        rm -rf chroma_db/ 2>/dev/null || true
        rm -rf *.db 2>/dev/null || true
        echo -e "${GREEN}✅ Generated data cleaned${NC}"
    else
        echo -e "${YELLOW}⏭️  Skipped data cleanup${NC}"
    fi
fi

echo
echo "=================================================="
echo "  ✅ Cleanup Complete!"
echo "=================================================="
