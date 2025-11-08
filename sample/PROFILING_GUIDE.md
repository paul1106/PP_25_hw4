# NCU Profiling Guide

## Quick Start

### Method 1: Using run_test.sh with NCU

```bash
# Normal test (no profiling)
./run_test.sh 00

# Run test with NCU profiling
./run_test.sh 00 --ncu
```

### Method 2: Using dedicated profile_ncu.sh

```bash
# Profile all kernels
./profile_ncu.sh 00

# Profile specific kernel only
./profile_ncu.sh 00 --kernel-name gpu_mine
```

### Method 3: Run in tmux (Recommended for Long Profiling)

```bash
# Start profiling in background tmux session
./run_ncu_tmux.sh 01

# Attach to session to view progress
tmux attach -t ncu_profile_case01

# Detach (keep running): Ctrl+B, then D
```

## Report Location

All NCU reports are saved in:
```
ncu_reports/
├── case00.ncu-rep
├── case01.ncu-rep
└── ...
```

## Viewing Reports

### Method 1: Download and Use Nsight Compute GUI (Recommended)

1. Download report file:
   ```bash
   scp <username>@cluster:~/hw4/sample/ncu_reports/case00.ncu-rep .
   ```

2. Open with NVIDIA Nsight Compute
   - Download from: https://developer.nvidia.com/nsight-compute

### Method 2: View Text Summary on Cluster

```bash
# View report summary
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 5 \
    bash -c "module load cuda && ncu --import ncu_reports/case00.ncu-rep"

# View specific kernel details
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 5 \
    bash -c "module load cuda && ncu --import ncu_reports/case00.ncu-rep --kernel-name gpu_mine"
```

## Important NCU Metrics

### Compute Performance
- **Compute (SM) Throughput**: GPU compute unit utilization
- **Memory Throughput**: Memory bandwidth utilization
- **Occupancy**: Thread occupancy rate

### Memory Access
- **Global Memory Access**: Global memory access count and efficiency
- **Shared Memory Access**: Shared memory usage
- **L1/L2 Cache Hit Rate**: Cache hit rates

### Execution Time
- **Duration**: Kernel execution time
- **Grid Size**: Grid and block configuration
- **Registers per Thread**: Register usage per thread

## Advanced NCU Options

### Collect Specific Metric Sets

```bash
# Memory metrics only (faster)
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --set memory --export ncu_reports/case00_memory \
    ./hw4 ../testcases/case00.in outputs/case00.out"

# Compute metrics only
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --set compute --export ncu_reports/case00_compute \
    ./hw4 ../testcases/case00.in outputs/case00.out"
```

### Profile Specific Kernel Launches

```bash
# Profile only first launch of gpu_mine kernel
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --kernel-name gpu_mine --launch-skip 0 --launch-count 1 \
    --export ncu_reports/case00_first_launch \
    ./hw4 ../testcases/case00.in outputs/case00.out"
```

## Understanding NCU Output

### Why So Many Passes?

When you see:
```
==PROF== Profiling "gpu_mine" - 0: 0%....50%....100% - 81 passes
==PROF== Profiling "gpu_mine" - 1: 0%....50%....100% - 81 passes
==PROF== Profiling "gpu_mine" - 2: 0%....50%....100% - 81 passes
```

**This is normal!**

- Each kernel launch requires **81 passes** to collect full metrics
- GPU hardware counters are limited - can't collect everything at once
- If your program launches `gpu_mine` 100 times → 100 × 81 = 8,100 total runs
- This is why profiling takes much longer than normal execution

### Time Estimation

Using `--set full` for complete metrics:

- **case00**: ~5-10 minutes (fewer kernel launches)
- **case01**: ~10-20 minutes (medium)
- **case02**: ~20-40 minutes (more kernel launches)
- **case03**: ~30-60 minutes (most kernel launches)

> **Note**: Each kernel launch needs 81 passes to collect complete performance metrics

## Tips

### 1. Why Use NCU?

- ✅ Avoids SSH disconnection issues during long profiling
- ✅ Can attach/detach to check progress anytime
- ✅ Safe for long-running profiling sessions

### 2. Speed Up Profiling

#### Option A: Profile Specific Kernel Only

```bash
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 15 \
    bash -c "module load cuda && ncu --set full --kernel-name gpu_mine --launch-count 5 \
    --export ncu_reports/case01_first5 \
    ./hw4 ../testcases/case01.in outputs/case01.out"
```

#### Option B: Collect Fewer Metrics

```bash
# Memory metrics only (faster, ~1/3 time)
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --set memory --export ncu_reports/case01_memory \
    ./hw4 ../testcases/case01.in outputs/case01.out"

# Compute metrics only
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --set compute --export ncu_reports/case01_compute \
    ./hw4 ../testcases/case01.in outputs/case01.out"
```

#### Option C: Use Basic Metrics

```bash
# Basic metrics (fastest)
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 5 \
    bash -c "module load cuda && ncu --set basic --launch-count 3 \
    --export ncu_reports/case00_quick \
    ./hw4 ../testcases/case00.in outputs/case00.out"
```

## Optimization Workflow

### 1. Baseline Test
```bash
./run_test.sh 00  # Record execution time
```

### 2. Collect Profiling Data
```bash
./profile_ncu.sh 00 --kernel-name gpu_mine
```

### 3. Analyze Bottlenecks
- Download `.ncu-rep` file
- Use Nsight Compute GUI to analyze
- Look at Memory Throughput, Compute Throughput, Occupancy

### 4. Optimize Code
- Modify `hw4.cu` based on bottlenecks
- Recompile: `make clean && make`

### 5. Verify Improvements
```bash
./run_test.sh 00              # Check correctness
./profile_ncu.sh 00 --kernel-name gpu_mine  # Compare metrics
```

### 6. Submit Results
```bash
hw4-judge  # Submit to scoreboard
```

## Key Performance Indicators

Focus on these metrics in NCU reports:

- **Compute (SM) Throughput**: Target > 80%
- **Memory Throughput**: Identify memory bottlenecks
- **Occupancy**: Thread utilization, recommend > 50%
- **L1/L2 Cache Hit Rate**: Cache efficiency
- **Warp Execution Efficiency**: SIMD efficiency
- **Register Usage**: Register pressure

## Troubleshooting

### Problem: NCU execution takes too long
**Solution**: Use smaller test case or limit profiling scope
```bash
./profile_ncu.sh 00 --kernel-name gpu_mine
```

### Problem: Report file too large
**Solution**: Collect specific metric sets only
```bash
srun -N 1 -n 1 --gpus-per-node 1 -A ACD114118 -t 10 \
    bash -c "module load cuda && ncu --set memory --export ncu_reports/case00_mem \
    ./hw4 ../testcases/case00.in outputs/case00.out"
```

### Problem: "ncu: No such file or directory"
**Solution**: Ensure CUDA module is loaded on compute node
```bash
srun ... bash -c "module load cuda && ncu ..."
```

### Problem: Session not found
**Solution**: List all sessions
```bash
tmux ls
# If no sessions, profiling may have completed or failed
ls -lh ncu_reports/
```

## tmux Commands Reference

```bash
# List all sessions
tmux ls

# Attach to session
tmux attach -t ncu_profile_case01

# Detach (keep running): Ctrl+B, then D

# Kill session
tmux kill-session -t ncu_profile_case01

# Scroll in tmux: Ctrl+B, [
# Exit scroll mode: Q
```

## Example Workflow

```bash
# 1. Start profiling
./run_ncu_tmux.sh 01

# 2. Wait a bit, check if it started
sleep 30
tmux attach -t ncu_profile_case01

# 3. Confirm it's running, then detach (Ctrl+B, D)

# 4. Do other work...

# 5. Check back later
tmux attach -t ncu_profile_case01

# 6. When complete, check report
ls -lh ncu_reports/case01.ncu-rep

# 7. Download to local machine
# (Run on local machine)
scp username@cluster:~/hw4/sample/ncu_reports/case01.ncu-rep .
```

## References

- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NCU Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUDA Profiling Best Practices](https://docs.nvidia.com/cuda/profiler-users-guide/)
