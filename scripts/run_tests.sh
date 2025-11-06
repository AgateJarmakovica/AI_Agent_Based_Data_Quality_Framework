#!/bin/bash
# Test runner script for healthdq-ai
# Author: Agate Jarmakoviča

set -e

echo "=================================================="
echo "  healthdq-ai - Test Runner"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
RUN_ALL=false
RUN_UNIT=false
RUN_INTEGRATION=false
RUN_COVERAGE=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --unit)
            RUN_UNIT=true
            shift
            ;;
        --integration)
            RUN_INTEGRATION=true
            shift
            ;;
        --coverage)
            RUN_COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --all          Run all tests"
            echo "  --unit         Run unit tests only"
            echo "  --integration  Run integration tests only"
            echo "  --coverage     Run with coverage report"
            echo "  --verbose, -v  Verbose output"
            echo "  --help, -h     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Default to all tests if nothing specified
if ! $RUN_UNIT && ! $RUN_INTEGRATION && ! $RUN_ALL; then
    RUN_ALL=true
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest not found. Installing...${NC}"
    pip install pytest pytest-cov pytest-asyncio
fi

# Verbose flag
PYTEST_ARGS=""
if $VERBOSE; then
    PYTEST_ARGS="-v"
fi

# Run tests
if $RUN_ALL || $RUN_UNIT; then
    echo "🧪 Running unit tests..."
    if $RUN_COVERAGE; then
        pytest tests/ $PYTEST_ARGS -m "not integration" --cov=src/healthdq --cov-report=term --cov-report=html
    else
        pytest tests/ $PYTEST_ARGS -m "not integration"
    fi
    echo -e "${GREEN}✅ Unit tests passed${NC}"
    echo
fi

if $RUN_ALL || $RUN_INTEGRATION; then
    echo "🔗 Running integration tests..."
    if $RUN_COVERAGE; then
        pytest tests/integration/ $PYTEST_ARGS --cov=src/healthdq --cov-report=term --cov-report=html --cov-append
    else
        pytest tests/integration/ $PYTEST_ARGS
    fi
    echo -e "${GREEN}✅ Integration tests passed${NC}"
    echo
fi

# Show coverage report location
if $RUN_COVERAGE; then
    echo "=================================================="
    echo "  📊 Coverage Report"
    echo "=================================================="
    echo
    echo "HTML report: htmlcov/index.html"
    echo "Open with: python -m http.server --directory htmlcov"
    echo
fi

echo "=================================================="
echo "  ✅ All Tests Passed!"
echo "=================================================="
