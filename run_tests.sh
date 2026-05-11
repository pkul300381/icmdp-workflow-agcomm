#!/bin/bash

# run_tests.sh
# -----------
# Script to execute unit tests and regression (smoke) tests for the ICMDP 
# Agentic Commerce application.
#
# Usage:
#   ./run_tests.sh          # Run all tests
#   ./run_tests.sh unit     # Run only unit tests
#   ./run_tests.sh regression # Run only regression tests

# Colors for pretty output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set the working directory to the project root
PROJECT_ROOT=$(dirname "$0")
cd "$PROJECT_ROOT"

# Default values
RUN_UNIT=true
RUN_REGRESSION=true

if [ "$1" == "unit" ]; then
    RUN_REGRESSION=false
elif [ "$1" == "regression" ]; then
    RUN_UNIT=false
fi

echo "=================================================="
echo "      ICMDP AGENTIC COMMERCE TEST RUNNER"
echo "=================================================="

# 1. Run Unit Tests using pytest
if [ "$RUN_UNIT" = true ] ; then
    echo -e "\n${BLUE}[1/2] Running Unit Tests (pytest)...${NC}"
    python3 -m pytest tests/
    UNIT_TEST_RESULT=$?
    
    if [ $UNIT_TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Unit Tests Passed!${NC}"
    else
        echo -e "${RED}✗ Unit Tests Failed!${NC}"
    fi
fi

# 2. Run Regression Test (Smoke test of main.py)
if [ "$RUN_REGRESSION" = true ] ; then
    echo -e "\n${BLUE}[2/2] Running Regression Test (Smoke test of main.py)...${NC}"
    echo "Running simulation to verify end-to-end pipeline..."
    python3 main.py > /dev/null 2>&1
    REGRESSION_RESULT=$?
    
    if [ $REGRESSION_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ Regression Test Passed! (main.py executed successfully)${NC}"
    else
        echo -e "${RED}✗ Regression Test Failed! (main.py encountered an error)${NC}"
    fi
fi

# Summary
echo -e "\n=================================================="
if ([ "$RUN_UNIT" = false ] || [ $UNIT_TEST_RESULT -eq 0 ]) && ([ "$RUN_REGRESSION" = false ] || [ $REGRESSION_RESULT -eq 0 ]); then
    echo -e "${GREEN}      ALL REQUESTED TESTS PASSED SUCCESSFULLY!${NC}"
    exit 0
else
    echo -e "${RED}      SOME TESTS FAILED. PLEASE CHECK LOGS ABOVE.${NC}"
    exit 1
fi
