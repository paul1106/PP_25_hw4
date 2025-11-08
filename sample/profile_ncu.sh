#!/bin/bash

# NCU Profiling Script
# Usage: ./profile_ncu.sh <case_number> [options]
# Example: ./profile_ncu.sh 00
#          ./profile_ncu.sh 00 --kernel-name gpu_mine

CASE=$1
KERNEL_NAME=""

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <case_number> [--kernel-name <name>]"
    echo "Example: $0 00"
    echo "         $0 00 --kernel-name gpu_mine  (profile specific kernel only)"
    exit 1
fi

shift
while [ $# -gt 0 ]; do
    case "$1" in
        --kernel-name)
            KERNEL_NAME="--kernel-name $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

INPUT="../testcases/case${CASE}.in"
OUTPUT="outputs/case${CASE}.out"
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
echo "NCU Profiling - case${CASE}"
echo "=========================================="
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "NCU Report: ${NCU_REPORT}.ncu-rep"
if [ -n "$KERNEL_NAME" ]; then
    echo "Target Kernel: $KERNEL_NAME"
fi
echo "=========================================="
echo ""

# Execute NCU profiling using srun
# --set full: collect comprehensive performance metrics
# --export: export report file
echo "Starting profiling (this may take a while)..."
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 15 \
    bash -c "module load cuda && ncu --set full $KERNEL_NAME --export ${NCU_REPORT} ./hw4 $INPUT $OUTPUT"

# Check if execution succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Profiling completed!"
    echo "Report saved to: ${NCU_REPORT}.ncu-rep"
    echo ""
    echo "To view the report:"
    echo "  1. Download report to local machine:"
    echo "     scp <username>@server:${PWD}/${NCU_REPORT}.ncu-rep ."
    echo ""
    echo "  2. Open with NVIDIA Nsight Compute UI"
    echo ""
    echo "  3. Or view summary in command line:"
    echo "     ncu --import ${NCU_REPORT}.ncu-rep"
    echo "=========================================="
    
    # Show file information
    if [ -f "${NCU_REPORT}.ncu-rep" ]; then
        ls -lh "${NCU_REPORT}.ncu-rep"
    fi
else
    echo "Profiling failed!"
    exit 1
fi
