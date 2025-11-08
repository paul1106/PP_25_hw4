# CUDA Bitcoin Mining - High Performance GPU Implementation

This project implements a CUDA-accelerated Bitcoin mining program, converting the original CPU version into a high-performance GPU version that leverages massive parallelism.

## Quick Start

```bash
# Load CUDA module
module load cuda

# Compile
make

# Run single test
./run_test.sh 00

# Run all tests
./run_all_tests.sh

# Submit to scoreboard
hw4-judge
```

## Project Structure

```
sample/
├── hw4                      # CUDA GPU version executable
├── hw4.cu                   # CUDA GPU source code (main version)
├── hw4_cpu.cu               # CPU version source code (backup)
├── sha256.cu / sha256.h     # SHA-256 implementation
├── Makefile                 # Build script
│
├── outputs/                 # Test output directory
├── ncu_reports/             # NCU profiling reports directory
│
├── run_test.sh              # Single test script (supports NCU)
├── run_all_tests.sh         # Batch test all cases
├── profile_ncu.sh           # Dedicated NCU profiling script
├── run_ncu_tmux.sh          # Run NCU in tmux session
│
└── Documentation:
    ├── CUDA_OPTIMIZATION.md       # CUDA optimization strategies
    ├── PROFILING_GUIDE.md         # NCU profiling guide
    └── TESTING_GUIDE.md           # Testing and debugging guide
```

## Key Features

### CUDA Optimizations Implemented

1. **Massive Parallelism**
   - 256 threads/block × 65,536 blocks = ~16.7 million threads
   - Each thread tests a unique nonce value

2. **Memory Hierarchy Optimization**
   - Constant memory for block header and SHA-256 constants
   - Register-based computation for hash variables
   - Minimized global memory access

3. **Instruction-Level Optimization**
   - Loop unrolling with `#pragma unroll`
   - Function inlining with `__forceinline__`
   - Optimized bit operations

4. **Early Termination**
   - Atomic operations for result coordination
   - Global flag for efficient thread termination

5. **Batch Execution Strategy**
   - Progressive nonce space exploration
   - Periodic result checking

### Expected Performance

- **CPU Version**: ~10^6 hashes/sec
- **GPU Version**: ~10^9+ hashes/sec
- **Speedup**: 100-1000x improvement

## Usage

### Basic Testing

```bash
# Test single case
./run_test.sh 00

# Test all cases
./run_all_tests.sh

# Compare with expected output
diff outputs/case00.out ../testcases/case00.out
```

### Performance Profiling

```bash
# Quick profiling (with test)
./run_test.sh 00 --ncu

# Detailed profiling
./profile_ncu.sh 00

# Profile specific kernel
./profile_ncu.sh 00 --kernel-name gpu_mine

# Run in tmux (recommended for long profiling)
./run_ncu_tmux.sh 01
tmux attach -t ncu_profile_case01
```

### Manual Execution

```bash
# Run on GPU node
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 4 \
    ./hw4 ../testcases/case00.in outputs/case00.out

# With NCU profiling
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 15 \
    bash -c "module load cuda && ncu --set full --export ncu_reports/case00 \
    ./hw4 ../testcases/case00.in outputs/case00.out"
```

## Documentation

- **CUDA_OPTIMIZATION.md** - Detailed optimization strategies and implementation
- **PROFILING_GUIDE.md** - Complete guide for NCU profiling
- **TESTING_GUIDE.md** - Testing procedures and debugging tips

## Compilation

```bash
make              # Compile hw4 (GPU version)
make hw4_cpu      # Compile hw4_cpu (CPU version, optional)
make clean        # Clean build artifacts
```

## Algorithm Overview

### SHA-256 Double Hashing

Bitcoin mining requires computing double SHA-256:
```
hash = SHA256(SHA256(block_header))
```

Where block_header contains:
- Version (4 bytes)
- Previous block hash (32 bytes)
- Merkle root (32 bytes)
- Timestamp (4 bytes)
- Difficulty bits (4 bytes)
- **Nonce (4 bytes)** ← What we're searching for

### Mining Process

1. Calculate Merkle root from transaction hashes
2. Prepare 80-byte block header
3. Test nonce values until hash < target difficulty
4. GPU parallelizes the nonce testing across millions of threads

## Performance Metrics

Key metrics to monitor in NCU reports:

- **Compute (SM) Throughput**: GPU utilization (target >80%)
- **Memory Throughput**: Memory bandwidth usage
- **Occupancy**: Thread utilization (recommend >50%)
- **L1/L2 Cache Hit Rate**: Cache efficiency
- **Warp Execution Efficiency**: SIMD efficiency
- **Register Usage**: Register pressure

## Troubleshooting

### Compilation Errors

```bash
# Ensure CUDA module is loaded
module load cuda

# Check nvcc version
nvcc --version

# Clean and rebuild
make clean && make
```

### Runtime Errors

```bash
# Check GPU availability
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 2 nvidia-smi

# Verify input files exist
ls -l ../testcases/

# Check output differences
diff outputs/case00.out ../testcases/case00.out
```

### Profiling Issues

```bash
# If NCU command not found, ensure CUDA is loaded on compute node
srun ... bash -c "module load cuda && ncu ..."

# If profiling takes too long, use smaller test case or limit launches
ncu --launch-count 5 ...
```

## References

- [SHA-256 Algorithm](https://en.wikipedia.org/wiki/SHA-2)
- [Bitcoin Block Hashing](https://en.bitcoin.it/wiki/Block_hashing_algorithm)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

## License

Academic project for Parallel Programming course.
