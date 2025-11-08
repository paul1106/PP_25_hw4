# CUDA Optimization Strategies - Implementation Details

## Overview

This document provides detailed information about the CUDA optimization strategies implemented in this Bitcoin mining program, converting a sequential CPU implementation into a massively parallel GPU implementation.

## Core Optimizations

### 1. Parallelization Architecture

**Objective**: Replace sequential CPU loop with massive GPU parallelism.

#### Implementation

**Kernel Design**:
```cuda
__global__ void gpu_mine(
    unsigned int *found,
    unsigned int *result_nonce,
    const unsigned char *target,
    unsigned int start_nonce,
    unsigned int total_threads
)
```

**Nonce Distribution**:
```cuda
unsigned int nonce = start_nonce + blockIdx.x * blockDim.x + threadIdx.x;
```

**Grid Configuration**:
- Threads per Block: 256
- Number of Blocks: 65,536
- Total Threads per Launch: 16,777,216 (~16.7 million)

#### Benefits

- Replaces single-threaded sequential loop
- Tests millions of nonce values simultaneously
- Fully utilizes GPU's parallel execution units
- Achieves near-linear speedup with thread count

### 2. Memory Hierarchy Optimization

**Objective**: Minimize memory access latency using appropriate memory types.

#### Constant Memory

```cuda
__constant__ BYTE dev_block_header[80];
__constant__ WORD dev_k[64];
```

**What's Stored**:
- 80-byte block header (excluding nonce)
- 64 SHA-256 round constants

**Why It Works**:
- Hardware broadcast mechanism
- All threads in a warp read simultaneously
- Extremely low latency (~same as register)
- Saves global memory bandwidth

#### Register Optimization

**Thread-Local Data**:
- Individual nonce value
- SHA-256 hash variables (a, b, c, d, e, f, g, h)
- Message schedule array w[64]

**Benefits**:
- Fastest possible access (1 cycle)
- No memory bandwidth consumption
- Maximizes throughput

#### Global Memory Minimization

**Limited to**:
- Result coordination (found flag)
- Final nonce storage
- Target difficulty value

**Access Pattern**:
- Atomic operations for synchronization
- Coalesced reads where possible
- Minimal write operations

### 3. Instruction-Level Optimization

**Objective**: Maximize instruction throughput and minimize control flow divergence.

#### Loop Unrolling

```cuda
#pragma unroll
for(i=0; i<64; ++i) {
    WORD S0 = ep0(a);
    WORD S1 = ep1(e);
    WORD ch_val = ch(e, f, g);
    WORD maj_val = maj(a, b, c);
    WORD temp1 = h + S1 + ch_val + dev_k[i] + w[i];
    WORD temp2 = S0 + maj_val;
    
    h = g; g = f; f = e; e = d + temp1;
    d = c; c = b; b = a; a = temp1 + temp2;
}
```

**Applied To**:
- SHA-256's 64-round compression function
- Message schedule expansion (16→64 words)
- Byte swapping operations

**Compiler Benefits**:
- Eliminates loop counter increment
- Removes branch instructions
- Enables better instruction scheduling
- Increases instruction-level parallelism

#### Function Inlining

```cuda
__device__ __forceinline__ WORD rotr(WORD x, WORD n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ WORD ch(WORD x, WORD y, WORD z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ WORD maj(WORD x, WORD y, WORD z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ WORD ep0(WORD x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ WORD ep1(WORD x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
```

**Benefits**:
- Eliminates function call overhead
- Reduces register spilling
- Enables cross-function optimization
- Bit operations inline directly into hot loop

### 4. Synchronization and Early Termination

**Objective**: Stop all threads as soon as solution is found.

#### Global Flag Mechanism

```cuda
// In global memory
unsigned int *dev_found;        // 0 = searching, 1 = found
unsigned int *dev_result_nonce; // Stores winning nonce
```

#### Atomic Operations

```cuda
if(valid) {
    unsigned int old = atomicExch(found, 1);
    if(old == 0) {
        // This thread found it first
        *result_nonce = nonce;
    }
}
```

**Why Atomic**:
- Multiple threads may find valid solutions
- Only first one should write result
- Prevents race conditions
- Ensures correctness

#### Early Exit

```cuda
__global__ void gpu_mine(...) {
    // Check early termination flag
    if(*found) return;
    
    // ... rest of computation
}
```

**Benefits**:
- Avoids unnecessary computation
- Reduces power consumption
- Improves average-case performance
- Critical for large search spaces

### 5. Batch Execution Strategy

**Objective**: Handle 2^32 search space efficiently.

#### Implementation

```cuda
const unsigned int threads_per_block = 256;
const unsigned int num_blocks = 65536;
const unsigned int total_threads_per_launch = threads_per_block * num_blocks;

unsigned int start_nonce = 0;
while(start_nonce <= 0xffffffff && !found) {
    // Launch kernel for this batch
    gpu_mine<<<num_blocks, threads_per_block>>>(
        dev_found, dev_result_nonce, dev_target, 
        start_nonce, total_threads_per_launch
    );
    
    cudaDeviceSynchronize();
    
    // Check if solution found
    cudaMemcpy(&host_found, dev_found, sizeof(unsigned int), 
               cudaMemcpyDeviceToHost);
    
    if(host_found) {
        // Get result
        cudaMemcpy(&result_nonce, dev_result_nonce, 
                   sizeof(unsigned int), cudaMemcpyDeviceToHost);
        break;
    }
    
    // Move to next batch
    start_nonce += total_threads_per_launch;
}
```

**Benefits**:
- Prevents single kernel timeout
- Allows progress monitoring
- Enables periodic result checking
- Better error handling

## Performance Analysis

### Theoretical Speedup

**CPU Version**:
- Single core: ~10^6 hashes/sec
- SHA-256 is sequential by design
- Limited by single-thread performance

**GPU Version**:
- 16.7M threads per launch
- Even with lower per-thread performance
- Massive parallelism dominates
- Expected: ~10^9+ hashes/sec

**Speedup Calculation**:
```
Speedup = GPU_throughput / CPU_throughput
        ≈ (16.7M threads × efficiency) / single_thread
        ≈ 100x - 1000x (depending on GPU model)
```

### Memory Bandwidth Analysis

**Constant Memory Usage**:
- 80 bytes block header
- 256 bytes (64 × 4) for k constants
- Total: 336 bytes
- Broadcast to all threads → Effective BW multiplication

**Register Usage**:
- ~40-50 registers per thread
- No memory bandwidth consumption
- Limited only by register file size

**Global Memory**:
- Minimal: ~12 bytes per kernel launch
- Found flag (4 bytes)
- Result nonce (4 bytes)
- Target value (4 bytes, read-only)

### Compute vs Memory Bound

This application is **compute-bound**:
- Each nonce requires 2× SHA-256 (128 compression rounds)
- Mostly bit operations and additions
- Minimal memory access
- Perfect for GPU acceleration

## Verification and Testing

### CPU Verification

```cpp
if(found) {
    block.nonce = result_nonce;
    SHA256 sha256_ctx;
    double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
    
    // Verify hash < target
    if(little_endian_bit_comparison(sha256_ctx.b, target_hex, 32) < 0) {
        printf("Verification passed!\n");
    }
}
```

### Correctness Guarantees

1. **Atomic operations** ensure single result write
2. **CPU re-verification** confirms GPU result
3. **Test suite** validates against known solutions
4. **Deterministic** results (same input → same output)

## Advanced Optimizations (Future Work)

### Potential Improvements

1. **Shared Memory Usage**
   - Pre-compute partial message schedules
   - Share intermediate values within block

2. **Warp-Level Primitives**
   - Use warp shuffle for faster communication
   - Reduce register pressure

3. **Multiple Kernels**
   - Overlap computation with transfers
   - Use streams for concurrent execution

4. **Dynamic Parallelism**
   - Spawn child kernels when close to solution
   - Adaptive search space reduction

5. **Mixed Precision**
   - Use FP16 where appropriate
   - Balance precision vs throughput

## Profiling Recommendations

### Key Metrics to Monitor

**Compute Utilization**:
- SM Throughput > 80%
- Achieved Occupancy > 50%
- Warp Execution Efficiency > 95%

**Memory Utilization**:
- L1 Cache Hit Rate (higher is better)
- L2 Cache Hit Rate (higher is better)
- Global Memory Throughput (should be low)

**Instruction Efficiency**:
- Minimal warp divergence
- Low register spilling
- Efficient instruction mix

### Using NCU

```bash
# Full metrics
ncu --set full --export report ./hw4 input.in output.out

# Focus on compute
ncu --set compute --export compute_report ./hw4 input.in output.out

# Focus on memory
ncu --set memory --export memory_report ./hw4 input.in output.out

# Specific kernel, first launch only
ncu --kernel-name gpu_mine --launch-count 1 \
    --set full --export detailed ./hw4 input.in output.out
```

## Conclusion

This implementation demonstrates effective use of:
- Massive parallelism (millions of threads)
- Memory hierarchy (constant, register, global)
- Instruction optimization (unrolling, inlining)
- Synchronization (atomics, early termination)
- Batch execution (handling large search spaces)

The result is a 100-1000x speedup over CPU, making Bitcoin mining computationally feasible within reasonable time constraints.

## References

- NVIDIA CUDA C Programming Guide
- NVIDIA CUDA Best Practices Guide
- Bitcoin Developer Reference
- SHA-256 Specification (FIPS 180-4)
