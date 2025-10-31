# GPU Context Pool Implementation Guide

**Quick Start:** Follow these exact steps to implement Option 2

---

## Prerequisites Check

```bash
# 1. Verify CRIU is built
ls -lh /root/criu/criu/criu
# Expected: 6.1M file exists

# 2. Verify CUDA plugin exists
ls -lh /root/criu/plugins/cuda/cuda_plugin.so
# Expected: 49K file exists

# 3. Verify CUDA headers available
ls /usr/local/cuda/include/cuda.h
# Expected: File exists

# 4. Current restore time baseline
time (podman stop vllm-llm-demo && podman container restore --keep vllm-llm-demo)
# Expected: ~5-6 seconds
```

---

## Step 1: Backup Original Files (2 minutes)

```bash
cd /root/criu/plugins/cuda

# Backup current cuda_plugin.c
cp cuda_plugin.c cuda_plugin.c.before-context-pool
cp cuda_plugin.c cuda_plugin.c.backup-$(date +%Y%m%d-%H%M%S)

# Verify backup
ls -lh cuda_plugin.c*
```

---

## Step 2: Apply Changes (10 minutes)

**Option A: Manual Editing (Recommended for understanding)**

Open the file:
```bash
cd /root/criu/plugins/cuda
vim cuda_plugin.c  # or nano, or any editor
```

Follow instructions in `/root/gpu-load/EXACT_CHANGES_CUDA_PLUGIN.patch`

Make 6 changes in order:
1. Add `#include <cuda.h>` after line 9
2. Add global variables after line 19
3. Add 3 new functions after line 570
4. Modify `cuda_plugin_init()` around line 626
5. Modify `resume_device()` around line 515
6. Modify `cuda_plugin_fini()` around line 657

**Option B: Automated Script (If you want speed)**

I can create a sed/awk script to apply changes automatically, but manual is safer for first time.

---

## Step 3: Verify Changes (2 minutes)

```bash
cd /root/criu/plugins/cuda

# Check line count increased
wc -l cuda_plugin.c
# Expected: ~840 lines (was 659)

# Check for new includes
head -15 cuda_plugin.c | grep "cuda.h"
# Expected: #include <cuda.h>

# Check for new functions
grep -n "init_gpu_context_pool" cuda_plugin.c
# Expected: Function definition found

# Check for context pool usage
grep -n "g_warm_context" cuda_plugin.c
# Expected: Multiple references
```

---

## Step 4: Build CRIU (3 minutes)

```bash
cd /root/criu

# Clean build to be safe
make clean

# Build (will take ~2 minutes)
make -j4

# Check for errors
echo $?
# Expected: 0 (success)

# Verify plugin rebuilt
ls -lh plugins/cuda/cuda_plugin.so
stat -c %y plugins/cuda/cuda_plugin.so  # Check timestamp is recent
```

---

## Step 5: Verify Symbols (1 minute)

```bash
# Check new functions are in the binary
nm /root/criu/plugins/cuda/cuda_plugin.so | grep -E "(warm|pool)"

# Expected output:
# 00000000000XXXXX t init_gpu_context_pool
# 00000000000XXXXX t fini_gpu_context_pool
# 00000000000XXXXX t prewarm_context_for_restore
# 00000000000XXXXX b g_warm_context
# 00000000000XXXXX b g_context_pool_enabled
# 00000000000XXXXX b g_context_lock
# 00000000000XXXXX b g_warm_device
# 00000000000XXXXX b g_context_initialized

# If you see these symbols, the build was successful!
```

---

## Step 6: Initial Test (2 minutes)

```bash
# Set environment variable to enable context pool
export CRIU_CUDA_CONTEXT_POOL=1

# Also enable parallel restore for maximum benefit
export CRIU_CUDA_PARALLEL_RESTORE=1
export CRIU_CUDA_STREAMS=8

# Stop container
podman stop vllm-llm-demo

# Restore with verbose output
podman container restore --keep vllm-llm-demo 2>&1 | tee /tmp/restore-test.log

# Check log for context pool messages
grep -i "context pool" /tmp/restore-test.log

# Expected messages:
#   "Initializing GPU context pool..."
#   "GPU context pool initialized successfully: ctx=0x... device=0"
#   "GPU context pool ENABLED - expect faster restore"
```

---

## Step 7: Quick Timing Test (5 minutes)

```bash
# Test WITHOUT context pool (baseline)
unset CRIU_CUDA_CONTEXT_POOL
echo "=== Test WITHOUT context pool ==="
for i in {1..3}; do
    podman stop vllm-llm-demo
    time podman container restore --keep vllm-llm-demo
    sleep 2
done

# Test WITH context pool
export CRIU_CUDA_CONTEXT_POOL=1
echo "=== Test WITH context pool ==="
for i in {1..3}; do
    podman stop vllm-llm-demo
    time podman container restore --keep vllm-llm-demo
    sleep 2
done

# Compare the "real" times
# Expected: WITH pool should be 0.3-0.7s faster
```

---

## Step 8: Full Benchmark (5 minutes)

```bash
cd /root/gpu-load

# Run comprehensive benchmark
./benchmark-criu-comparison.py

# Expected output:
# ┌───────────────────────┬─────────────────┬─────────┬─────────┬──────────┐
# │ CRIU                  │ Avg Restore (s) │ Min (s) │ Max (s) │  Speedup │
# ├───────────────────────┼─────────────────┼─────────┼─────────┼──────────┤
# │ Without context pool  │           5.688 │   5.631 │   5.723 │     1.00x│
# │ With context pool     │           5.000 │   4.950 │   5.050 │     1.14x│
# └───────────────────────┴─────────────────┴─────────┴─────────┴──────────┘
#
# Improvement: 0.688s faster (12% improvement)
```

---

## Troubleshooting

### Build Fails with "cuda.h not found"

```bash
# Find CUDA installation
find /usr/local -name "cuda.h" 2>/dev/null

# If found, update Makefile
cd /root/criu/plugins/cuda
# Edit Makefile and add: -I/path/to/cuda/include

# Or symlink
sudo ln -s /usr/local/cuda-*/include /usr/local/cuda/include
```

### Context Pool Not Initializing

```bash
# Check environment variable is set
echo $CRIU_CUDA_CONTEXT_POOL
# Expected: 1

# Check CRIU logs
grep "context pool" /tmp/restore-test.log

# If "cuInit failed", check GPU availability
nvidia-smi

# If driver/device mismatch, may need to restart nvidia-persistenced
sudo systemctl restart nvidia-persistenced
```

### No Performance Improvement

```bash
# Verify warm context is being used
grep "Warm context set" /tmp/restore-test.log

# If not found, context may not be warming properly
# Check if cuda-checkpoint is respecting the existing context
# May need LD_PRELOAD approach instead (next step)
```

### Container Fails to Restore

```bash
# Disable context pool immediately
unset CRIU_CUDA_CONTEXT_POOL

# Restore should work normally
podman container restore --keep vllm-llm-demo

# If works without pool, there may be a compatibility issue
# File an issue with logs
```

---

## Rollback (if needed)

```bash
# Restore original file
cd /root/criu/plugins/cuda
cp cuda_plugin.c.before-context-pool cuda_plugin.c

# Rebuild
cd /root/criu
make clean && make

# Verify original behavior
podman container restore --keep vllm-llm-demo
```

---

## Success Criteria

✅ Build succeeds without errors
✅ Context pool initializes (log message confirms)
✅ Restore still works correctly
✅ 0.3-0.7s performance improvement measured
✅ Multiple restores work (no memory leaks)

---

## Next Steps (After Success)

1. **Document the improvement** in project notes
2. **Monitor stability** over multiple restores
3. **Consider LD_PRELOAD** for additional 1.5s improvement
4. **Share results** if performance goals met

---

## Quick Reference

**Files:**
- State: `/root/gpu-load/CURRENT_STATE_2025-10-29.md`
- Patch: `/root/gpu-load/EXACT_CHANGES_CUDA_PLUGIN.patch`
- This guide: `/root/gpu-load/IMPLEMENTATION_GUIDE.md`

**Environment Variables:**
```bash
export CRIU_CUDA_CONTEXT_POOL=1        # Enable context pool
export CRIU_CUDA_PARALLEL_RESTORE=1    # Enable parallel restore
export CRIU_CUDA_STREAMS=8             # Number of streams
export CRIU_CUDA_CHUNK_MB=256          # Chunk size
```

**Test Command:**
```bash
time (podman stop vllm-llm-demo && podman container restore --keep vllm-llm-demo)
```

**Expected Timeline:**
- Backup & apply changes: 15 minutes
- Build & test: 10 minutes
- Benchmark & validate: 10 minutes
- **Total: 35 minutes to completion**

---

**READY TO START!**

Begin with Step 1 (Backup) and follow each step in order.
