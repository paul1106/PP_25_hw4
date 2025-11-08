# Testing Guide

## Quick Start

### Run Single Test

```bash
# Test case 00
./run_test.sh 00

# Test case 01
./run_test.sh 01
```

### Run All Tests

```bash
./run_all_tests.sh
```

### Test with Profiling

```bash
./run_test.sh 00 --ncu
```

## Test Case Structure

```
testcases/
├── case00.in   → Input file (block header + target)
├── case00.out  → Expected output (nonce)
├── case01.in
├── case01.out
├── case02.in
├── case02.out
├── case03.in
└── case03.out
```

## Understanding Test Cases

Each test case consists of:

1. **Input file (`.in`)**:
   - Line 1: 80-byte block header (hex)
   - Line 2: 32-byte target difficulty (hex)

2. **Output file (`.out`)**:
   - Single number: valid nonce value

## Test Execution Methods

### Method 1: Using run_test.sh (Recommended)

```bash
# Basic test
./run_test.sh 00

# With profiling
./run_test.sh 00 --ncu

# On compute node with specific time limit
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 ./run_test.sh 00
```

### Method 2: Manual Execution

```bash
# 1. Compile
make clean && make

# 2. Run
./hw4 ../testcases/case00.in outputs/case00.out

# 3. Verify
diff outputs/case00.out ../testcases/case00.out
```

### Method 3: Batch Testing

```bash
./run_all_tests.sh
```

## Verification

### Automatic Verification (via run_test.sh)

The script automatically compares output with expected result:

```
==================================================
Testing case 00
==================================================
Running: ./hw4 ../testcases/case00.in outputs/case00.out
Execution time: 0.234s
Checking result...
✓ PASS: Output matches expected result
==================================================
```

### Manual Verification

```bash
# Check if output matches expected
diff outputs/case00.out ../testcases/case00.out

# No output = files match (test passed)
# Output shown = files differ (test failed)
```

### Verification with Details

```bash
# Show difference if any
diff -u outputs/case00.out ../testcases/case00.out

# Compare byte by byte
cmp outputs/case00.out ../testcases/case00.out
```

## Test Case Characteristics

### case00: Small (Quick)
- **Difficulty**: Low
- **Nonce Range**: Small search space
- **Execution Time**: < 1 second
- **Purpose**: Quick correctness check
- **Use Case**: Rapid development iteration

### case01: Medium
- **Difficulty**: Moderate
- **Nonce Range**: Medium search space
- **Execution Time**: ~1-5 seconds
- **Purpose**: Performance validation
- **Use Case**: Initial performance testing

### case02: Large
- **Difficulty**: High
- **Nonce Range**: Large search space
- **Execution Time**: ~5-30 seconds
- **Purpose**: Stress testing
- **Use Case**: Verify batch execution logic

### case03: Very Large
- **Difficulty**: Very High
- **Nonce Range**: Very large search space
- **Execution Time**: ~30-120 seconds
- **Purpose**: Full-scale benchmark
- **Use Case**: Final performance validation

## Debugging Failed Tests

### Step 1: Check Compilation

```bash
make clean && make
```

Look for warnings or errors.

### Step 2: Run with Debug Output

Modify `hw4.cu` temporarily:

```cuda
// Add after finding nonce
if (found_nonce != 0) {
    printf("DEBUG: Found nonce = %u\n", found_nonce);
    
    // Verify hash
    BYTE hash[SHA256_BLOCK_SIZE];
    sha256_hash_block(block_header, found_nonce, hash);
    printf("DEBUG: Hash = ");
    for (int i = 0; i < SHA256_BLOCK_SIZE; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
}
```

### Step 3: Verify Nonce Value

```bash
# Get expected nonce
cat ../testcases/case00.out

# Get your output
cat outputs/case00.out

# Compare
diff outputs/case00.out ../testcases/case00.out
```

### Step 4: Test CPU Version

```bash
# Compile CPU version
make hw4_cpu

# Run on same input
./hw4_cpu ../testcases/case00.in outputs/case00_cpu.out

# Compare with GPU output
diff outputs/case00.out outputs/case00_cpu.out

# Compare with expected
diff outputs/case00_cpu.out ../testcases/case00.out
```

### Step 5: Check for Common Issues

#### Issue 1: Wrong Endianness
```cuda
// Ensure proper byte order conversion
// SHA-256 uses big-endian
```

#### Issue 2: Incomplete Search Space
```cuda
// Verify batch execution covers [0, 2^32)
// Check that all nonce values are tested
```

#### Issue 3: Race Conditions
```cuda
// Ensure atomic operations for shared state
atomicExch(found, 1);
```

#### Issue 4: Early Termination Logic
```cuda
// Verify early exit doesn't skip valid nonce
if (*found == 1 && found_nonce != 0) {
    break;  // Only break if valid nonce found
}
```

## Performance Testing

### Measure Execution Time

```bash
# Using time command
time ./hw4 ../testcases/case01.in outputs/case01.out

# Using run_test.sh (includes timing)
./run_test.sh 01
```

### Compare CPU vs GPU

```bash
# Compile both versions
make hw4_cpu
make

# Time CPU version
time ./hw4_cpu ../testcases/case01.in outputs/case01_cpu.out

# Time GPU version
time ./hw4 ../testcases/case01.in outputs/case01.out

# Calculate speedup
```

### Expected Performance

| Test Case | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| case00    | ~1s      | ~0.01s   | ~100x   |
| case01    | ~10s     | ~0.1s    | ~100x   |
| case02    | ~100s    | ~0.5s    | ~200x   |
| case03    | ~1000s   | ~2s      | ~500x   |

> **Note**: Actual times depend on hardware specifications

## Cluster Execution

### Interactive Testing

```bash
# Request GPU node for 10 minutes
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 --pty bash

# Now on compute node - run tests
cd ~/hw4/sample
./run_test.sh 00
./run_test.sh 01
```

### Batch Job Testing

Create `test_job.sh`:

```bash
#!/bin/bash
#SBATCH -J hw4_test
#SBATCH -A ACD114118
#SBATCH -p gtest
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:15:00
#SBATCH -o test_%j.log

module load cuda

cd ~/hw4/sample
./run_all_tests.sh
```

Submit:

```bash
sbatch test_job.sh
```

### One-liner Testing

```bash
# Run single test
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "cd ~/hw4/sample && ./run_test.sh 00"

# Run all tests
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 15 \
    bash -c "cd ~/hw4/sample && ./run_all_tests.sh"
```

## Test Output Organization

```
outputs/
├── case00.out       # Your GPU output
├── case01.out
├── case02.out
├── case03.out
└── case00_cpu.out   # CPU version output (for comparison)
```

## Troubleshooting

### Problem: "No such file or directory"

**Cause**: Missing test case files or incorrect path

**Solution**:
```bash
# Verify test case exists
ls -l ../testcases/case00.in
ls -l ../testcases/case00.out

# Check current directory
pwd
# Should be: /home/username/hw4/sample
```

### Problem: "Permission denied"

**Cause**: Script not executable

**Solution**:
```bash
chmod +x run_test.sh
chmod +x run_all_tests.sh
chmod +x profile_ncu.sh
chmod +x run_ncu_tmux.sh
```

### Problem: Output differs from expected

**Cause**: Implementation error

**Solution**:
1. Check SHA-256 implementation
2. Verify nonce increment logic
3. Test CPU version for comparison
4. Add debug output to trace execution

### Problem: Program doesn't terminate

**Cause**: 
- Infinite loop in batch execution
- Early termination not working
- Missing nonce in search space

**Solution**:
```bash
# Kill the process
Ctrl+C

# Check batch execution logic
# Verify loop termination condition
# Ensure full [0, 2^32) range is covered
```

### Problem: Incorrect on large test cases only

**Cause**: 
- Batch execution not covering full space
- Integer overflow in nonce calculation
- Race condition in parallel execution

**Solution**:
```cuda
// Use proper data types
uint32_t start_nonce = ...;
uint32_t total_threads = ...;

// Check overflow
if (start_nonce + total_threads < start_nonce) {
    // Handle overflow
}
```

## Validation Checklist

Before submission:

- [ ] All test cases pass (case00-case03)
- [ ] GPU output matches expected output
- [ ] CPU version gives same result (if available)
- [ ] No compilation warnings
- [ ] Execution time is reasonable
- [ ] Code compiles with `make`
- [ ] No hardcoded values (except constants)
- [ ] Proper error handling

## Best Practices

1. **Test Frequently**: Run quick test (case00) after each change
2. **Use Small Cases First**: Debug with case00, then scale to larger cases
3. **Compare with CPU**: Use CPU version as ground truth
4. **Check Intermediate Values**: Add debug output for verification
5. **Test Edge Cases**: Verify nonce = 0, nonce = 2^32-1
6. **Clean Build**: Use `make clean && make` when in doubt

## hw4-judge Submission

After all local tests pass:

```bash
# Final verification
./run_all_tests.sh

# Submit to judge
hw4-judge

# Check scoreboard
hw4-scoreboard
```

## References

- [Bitcoin Mining Overview](../CUDA_OPTIMIZATION.md#bitcoin-mining-algorithm)
- [Profiling Guide](PROFILING_GUIDE.md)
- [CUDA Optimizations](CUDA_OPTIMIZATION.md)
