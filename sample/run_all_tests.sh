#!/bin/bash

# Batch test all test cases
# Usage: ./run_all_tests.sh

echo "=========================================="
echo "Batch Testing CUDA Bitcoin Mining"
echo "=========================================="
echo ""

# Ensure output directory exists
mkdir -p outputs/

# Test case numbers
CASES=("00" "01" "02" "03")

for CASE in "${CASES[@]}"; do
    INPUT="../testcases/case${CASE}.in"
    
    # Check if input file exists
    if [ ! -f "$INPUT" ]; then
        echo "Skipping case${CASE} (file not found)"
        continue
    fi
    
    echo "Running case${CASE}..."
    ./run_test.sh $CASE
    echo ""
    echo "=========================================="
    echo ""
done

echo "All tests completed!"
