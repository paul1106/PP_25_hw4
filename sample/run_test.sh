#!/bin/bash

# Test CUDA Bitcoin Mining Program
# Usage: ./run_test.sh <case_number> [--ncu]
# Example: ./run_test.sh 00
#          ./run_test.sh 00 --ncu  (generate NCU profiling report)

ENABLE_NCU=0

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <case_number> [--ncu]"
    echo "Example: $0 00"
    echo "         $0 00 --ncu  (generate NCU profiling report)"
    exit 1
fi

CASE=$1

# Check if NCU profiling is enabled
if [ "$2" == "--ncu" ]; then
    ENABLE_NCU=1
fi

INPUT="../testcases/case${CASE}.in"
OUTPUT="outputs/case${CASE}.out"
EXPECTED="../testcases/case${CASE}.out"
NCU_REPORT="ncu_reports/case${CASE}"

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

# Ensure output directories exist
mkdir -p outputs/
mkdir -p ncu_reports/

echo "=========================================="
echo "Test case: case${CASE}"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
if [ $ENABLE_NCU -eq 1 ]; then
    echo "NCU Report: ${NCU_REPORT}.ncu-rep"
fi
echo "=========================================="

# Execute on GPU node using srun
if [ $ENABLE_NCU -eq 1 ]; then
    echo "Enabling NCU Profiling..."
    srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
        bash -c "module load cuda && ncu --set full --export ${NCU_REPORT} ./hw4 $INPUT $OUTPUT"
else
    srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 4 ./hw4 $INPUT $OUTPUT
fi

# Check if execution succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Execution completed!"
    
    # If expected output exists, compare
    if [ -f "$EXPECTED" ]; then
        echo ""
        echo "Comparing results..."
        if diff -q $OUTPUT $EXPECTED > /dev/null; then
            echo "✓ Test PASSED! Output matches expected result"
        else
            echo "✗ Test FAILED! Output differs from expected"
            echo ""
            echo "Differences:"
            diff $OUTPUT $EXPECTED
        fi
    fi
else
    echo "Execution failed!"
    exit 1
fi
