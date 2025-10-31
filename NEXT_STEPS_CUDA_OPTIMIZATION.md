# CUDA Checkpoint Optimization: Actionable Implementation Plan
## Immediate Next Steps for 5-10× Restore Speedup

**Date**: 2025-10-29
**Priority**: HIGH
**Target**: 39s → 7s restore time for LLaMA 3.1 8B

---

## Quick Reference

### Current Situation
- **Baseline**: 38.8s restore time (28s CPU + 11s GPU)
- **Bottleneck**: Single-threaded operations in both CPU and GPU restore
- **Root Cause**: NVIDIA's cuda-checkpoint is inherently single-threaded
- **Your Code Status**: ✅ Parallel restore infrastructure already built, just needs integration

### What This Document Provides
1. Step-by-step implementation guides
2. Exact code changes with line numbers
3. Testing procedures
4. Expected performance improvements
5. Risk mitigation strategies

---

## Phase 1: GPU Context Pooling (IMMEDIATE - 1-2 Days)

### Why This First?
- ✅ Lowest risk (no cuda-checkpoint modification)
- ✅ Immediate 8% speedup (2.9s savings)
- ✅ Simple code changes (10-20 lines)
- ✅ No external dependencies

### Performance Impact
```
Context Creation: 3.1s → 0.2s (15× faster)
Total Restore:    38.8s → 35.9s (8% faster)
```

### Implementation

#### Step 1.1: Modify cuda_plugin_init()
**File**: `/root/criu/plugins/cuda/cuda_plugin.c`
**Location**: Function `cuda_plugin_init()`, line ~617

**Add this code**:
```c
// In the RESTORE stage initialize parallel GPU memory restore
if (stage == CR_PLUGIN_STAGE__RESTORE) {
    cuda_parallel_config_t config;
    cuda_parallel_config_from_env(&config);

    if (cuda_parallel_restore_init(&config) < 0) {
        pr_warn("Failed to initialize CUDA parallel restore, using standard path\n");
        /* Continue with standard restore - not a fatal error */
    }

    // *** NEW: Pre-warm GPU contexts ***
    if (cuda_context_pool_init() < 0) {
        pr_warn("Failed to initialize GPU context pool, contexts will be created on-demand\n");
    }
}
```

#### Step 1.2: Add Context Pool Functions
**File**: `/root/criu/plugins/cuda/cuda_parallel_restore.c`
**Location**: Add before `cuda_parallel_restore_memory()` function

```c
/* Global context pool */
static CUcontext *g_context_pool = NULL;
static int g_context_pool_size = 0;

/**
 * Initialize GPU context pool.
 * Pre-creates CUDA contexts for all available GPUs to eliminate
 * context creation overhead during restore.
 */
int cuda_context_pool_init(void)
{
    cudaError_t err;
    int device_count = 0;

    pr_info("Initializing GPU context pool\n");

    /* Get device count */
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        pr_err("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (device_count == 0) {
        pr_warn("No CUDA devices found for context pool\n");
        return 0;
    }

    pr_info("Found %d CUDA devices, pre-warming contexts\n", device_count);

    /* Allocate context pool */
    g_context_pool = malloc(device_count * sizeof(CUcontext));
    if (!g_context_pool) {
        pr_perror("Failed to allocate context pool");
        return -1;
    }

    /* Pre-initialize each device */
    for (int i = 0; i < device_count; i++) {
        err = cudaSetDevice(i);
        if (err != cudaSuccess) {
            pr_err("cudaSetDevice(%d) failed: %s\n", i, cudaGetErrorString(err));
            goto cleanup;
        }

        /* Force CUDA runtime initialization for this device */
        err = cudaFree(0);
        if (err != cudaSuccess) {
            pr_err("cudaFree(0) failed for device %d: %s\n", i, cudaGetErrorString(err));
            goto cleanup;
        }

        pr_info("Pre-warmed context for device %d\n", i);
    }

    g_context_pool_size = device_count;
    pr_info("GPU context pool initialized with %d contexts\n", device_count);
    return 0;

cleanup:
    free(g_context_pool);
    g_context_pool = NULL;
    g_context_pool_size = 0;
    return -1;
}

/**
 * Cleanup context pool.
 */
void cuda_context_pool_fini(void)
{
    if (g_context_pool) {
        /* Reset device (contexts are automatically cleaned up) */
        cudaDeviceReset();
        free(g_context_pool);
        g_context_pool = NULL;
        g_context_pool_size = 0;
        pr_info("GPU context pool cleaned up\n");
    }
}
```

#### Step 1.3: Update Header File
**File**: `/root/criu/plugins/cuda/cuda_parallel_restore.h`
**Location**: Add before closing `#endif`

```c
/*
 * Initialize GPU context pool for faster restore.
 * Pre-warms CUDA contexts to eliminate initialization overhead.
 * Must be called during CRIU plugin initialization (restore stage).
 *
 * Returns: 0 on success, -1 on failure (non-fatal)
 */
int cuda_context_pool_init(void);

/*
 * Cleanup GPU context pool.
 * Must be called during plugin finalization.
 */
void cuda_context_pool_fini(void);
```

#### Step 1.4: Update Plugin Finalization
**File**: `/root/criu/plugins/cuda/cuda_plugin.c`
**Location**: Function `cuda_plugin_fini()`, line ~654

**Add before `cuda_parallel_restore_fini()`**:
```c
/* In the RESTORE stage cleanup parallel GPU memory restore */
if (stage == CR_PLUGIN_STAGE__RESTORE) {
    cuda_context_pool_fini();  // *** NEW ***
    cuda_parallel_restore_fini();
}
```

### Testing

#### Test 1.1: Verify Context Pool Initialization
```bash
# Enable debug logging
export CRIU_DEBUG=1

# Run restore with context pooling
criu restore --images-dir /checkpoint --shell-job -v4 2>&1 | grep -i "context pool"

# Expected output:
# cuda_parallel: Initializing GPU context pool
# cuda_parallel: Found 1 CUDA devices, pre-warming contexts
# cuda_parallel: Pre-warmed context for device 0
# cuda_parallel: GPU context pool initialized with 1 contexts
```

#### Test 1.2: Measure Performance Improvement
```bash
# Baseline (before changes)
time criu restore --images-dir /checkpoint --shell-job

# With context pooling
time criu restore --images-dir /checkpoint --shell-job

# Compare GPU context creation time in logs
grep "Context creation" /var/log/criu/restore.log
```

### Success Criteria
- ✅ Context pool initializes without errors
- ✅ All GPUs show "Pre-warmed context" messages
- ✅ Restore time decreases by 2-3 seconds
- ✅ No crashes or hangs during restore

---

## Phase 2: LD_PRELOAD Interception (HIGH PRIORITY - 2-3 Weeks)

### Why This Second?
- ✅ Highest GPU-specific impact (5.5× GPU speedup)
- ✅ Reuses your existing parallel infrastructure
- ✅ No modifications to cuda-checkpoint binary
- ⚠️ Medium risk (must not break normal apps)

### Performance Impact
```
GPU Memory Transfer: 7.9s → 1.8s (4.4× faster)
Total Restore:       35.9s → 30.0s (20% additional speedup)
```

### Implementation

#### Step 2.1: Create Interception Library
**New File**: `/root/criu/plugins/cuda/cuda_intercept.c`

```c
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdbool.h>
#include <cuda_runtime.h>

#include "cuda_parallel_restore.h"

/* Restore phase tracking */
static __thread bool in_restore_phase = false;

/* Original function pointers */
static cudaError_t (*real_cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind) = NULL;
static cudaError_t (*real_cudaMemcpyAsync)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;

/* Minimum size to trigger parallel transfer (256 MB) */
#define MIN_PARALLEL_SIZE (256 * 1024 * 1024)

/* Initialize function pointers */
static void __attribute__((constructor)) init_intercept(void)
{
    real_cudaMemcpy = dlsym(RTLD_NEXT, "cudaMemcpy");
    real_cudaMemcpyAsync = dlsym(RTLD_NEXT, "cudaMemcpyAsync");

    if (!real_cudaMemcpy || !real_cudaMemcpyAsync) {
        fprintf(stderr, "cuda_intercept: Failed to load original CUDA functions\n");
        return;
    }

    const char *env = getenv("CUDA_INTERCEPT_RESTORE");
    if (env && atoi(env) == 1) {
        in_restore_phase = true;
        fprintf(stderr, "cuda_intercept: Restore phase active\n");
    }
}

/*
 * Intercept cudaMemcpy - redirect large D2H transfers to parallel implementation
 */
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    if (!real_cudaMemcpy) {
        fprintf(stderr, "cuda_intercept: cudaMemcpy not initialized\n");
        return cudaErrorInitializationError;
    }

    /* Only intercept device-to-host transfers during restore */
    if (in_restore_phase && kind == cudaMemcpyDeviceToHost && count >= MIN_PARALLEL_SIZE) {
        fprintf(stderr, "cuda_intercept: Intercepting cudaMemcpy D2H: %zu MB\n", count / (1024*1024));

        /* Use parallel restore infrastructure */
        int ret = cuda_parallel_restore_buffer((void*)src, dst, count);
        if (ret == 0) {
            return cudaSuccess;
        }

        /* Fallback to original on error */
        fprintf(stderr, "cuda_intercept: Parallel restore failed, using fallback\n");
    }

    return real_cudaMemcpy(dst, src, count, kind);
}

/*
 * Intercept cudaMemcpyAsync - same logic as synchronous version
 */
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream)
{
    if (!real_cudaMemcpyAsync) {
        fprintf(stderr, "cuda_intercept: cudaMemcpyAsync not initialized\n");
        return cudaErrorInitializationError;
    }

    if (in_restore_phase && kind == cudaMemcpyDeviceToHost && count >= MIN_PARALLEL_SIZE) {
        fprintf(stderr, "cuda_intercept: Intercepting cudaMemcpyAsync D2H: %zu MB\n", count / (1024*1024));

        int ret = cuda_parallel_restore_buffer((void*)src, dst, count);
        if (ret == 0) {
            return cudaSuccess;
        }

        fprintf(stderr, "cuda_intercept: Parallel restore failed, using fallback\n");
    }

    return real_cudaMemcpyAsync(dst, src, count, kind, stream);
}
```

#### Step 2.2: Build Shared Library
**File**: `/root/criu/plugins/cuda/Makefile` (modify existing)

**Add these rules**:
```makefile
# Existing rules...

# New interception library
libcuda_intercept.so: cuda_intercept.c cuda_parallel_restore.c
	$(CC) -shared -fPIC -O2 -Wall \
	    -I/usr/local/cuda/include \
	    -L/usr/local/cuda/lib64 \
	    -o $@ $^ \
	    -lcuda -lcudart -ldl -lpthread

all: cuda_plugin.so libcuda_intercept.so

clean:
	rm -f *.so *.o

install: all
	cp cuda_plugin.so /usr/lib/criu/
	cp libcuda_intercept.so /usr/local/lib/
	ldconfig
```

#### Step 2.3: Integrate with CRIU Plugin
**File**: `/root/criu/plugins/cuda/cuda_plugin.c`
**Location**: Function `cuda_plugin_resume_devices_late()`, line ~543

**Modify to set environment variable**:
```c
int cuda_plugin_resume_devices_late(int pid)
{
    if (plugin_disabled) {
        return -ENOTSUP;
    }

    /* *** NEW: Enable CUDA interception for parallel restore *** */
    char *old_intercept = getenv("CUDA_INTERCEPT_RESTORE");
    setenv("CUDA_INTERCEPT_RESTORE", "1", 1);

    /* RESUME_DEVICES_LATE is used during `criu restore`.
     * Here, we assume that users expect the target process
     * to be in a "running" state after restore, even if it was
     * in a "locked" or "checkpointed" state during `criu dump`.
     */
    int ret = resume_device(pid, 1, CUDA_TASK_RUNNING);

    /* *** NEW: Restore original environment *** */
    if (old_intercept) {
        setenv("CUDA_INTERCEPT_RESTORE", old_intercept, 1);
    } else {
        unsetenv("CUDA_INTERCEPT_RESTORE");
    }

    return ret;
}
```

#### Step 2.4: Update cuda_parallel_restore_buffer()
**File**: `/root/criu/plugins/cuda/cuda_parallel_restore.c`
**Location**: Function `cuda_parallel_restore_buffer()`, line ~334

**Make function accessible to intercept library** (already implemented, just verify):
```c
/*
 * Utility function to restore memory if device and host pointers are known.
 * This can be called from LD_PRELOAD hooks or future integrations.
 */
int cuda_parallel_restore_buffer(void *device_ptr, void *host_ptr, size_t size)
{
    // ... existing implementation is correct ...
    return ret;
}
```

### Testing

#### Test 2.1: Verify Interception Works
```bash
# Build the interception library
cd /root/criu/plugins/cuda
make libcuda_intercept.so

# Test with LD_PRELOAD
export CUDA_INTERCEPT_RESTORE=1
export LD_PRELOAD=/root/criu/plugins/cuda/libcuda_intercept.so

# Run cuda-checkpoint manually
cuda-checkpoint --action restore --pid $PID

# Check output for interception messages:
# cuda_intercept: Restore phase active
# cuda_intercept: Intercepting cudaMemcpy D2H: 1024 MB
```

#### Test 2.2: Integration Test with CRIU
```bash
# Create checkpoint
podman container checkpoint --export=/tmp/test.tar.gz test-container

# Restore with interception enabled
export LD_PRELOAD=/root/criu/plugins/cuda/libcuda_intercept.so
time criu restore --images-dir /checkpoint --shell-job -v4

# Verify in logs:
grep "Intercepting cudaMemcpy" /var/log/criu/restore.log
grep "Parallel restore" /var/log/criu/restore.log
```

#### Test 2.3: Performance Benchmark
```bash
# Baseline (no interception)
unset LD_PRELOAD
time criu restore --images-dir /checkpoint --shell-job > baseline.log 2>&1

# With interception
export LD_PRELOAD=/root/criu/plugins/cuda/libcuda_intercept.so
time criu restore --images-dir /checkpoint --shell-job > intercept.log 2>&1

# Compare GPU restore times
grep "GPU state restore" baseline.log intercept.log
```

### Success Criteria
- ✅ Library builds without errors
- ✅ Interception messages appear in logs
- ✅ GPU memory transfer time decreases by 60-75%
- ✅ No crashes or data corruption
- ✅ Application runs correctly after restore

---

## Phase 3: CPU Memory Parallelization (2-4 Weeks)

### Reference
See detailed implementation in: `/root/gpu-load/CRIU_RESTORE_IMPROVEMENTS_IMPLEMENTATION.md`

### Summary
- Multi-threaded preadv() in restorer.c
- io_uring for async I/O
- Parallel process tree restoration

### Expected Impact
```
CPU Memory Restore: 28s → 5s (5.6× faster)
Combined Total:     30s → 7s (4.3× faster than baseline)
```

---

## Environment Variables Reference

### CUDA Parallel Restore Configuration
```bash
# Enable/disable parallel restore (default: 1)
export CRIU_CUDA_PARALLEL_RESTORE=1

# Number of CUDA streams (default: 4, range: 1-32)
export CRIU_CUDA_STREAMS=8

# Chunk size in MB (default: 256)
export CRIU_CUDA_CHUNK_MB=512

# Use pinned memory for faster transfers (default: 1)
export CRIU_CUDA_USE_PINNED_MEM=1
```

### CUDA Interception Control
```bash
# Enable LD_PRELOAD interception (set by plugin automatically)
export CUDA_INTERCEPT_RESTORE=1

# Preload the interception library
export LD_PRELOAD=/root/criu/plugins/cuda/libcuda_intercept.so
```

### Optimal Configuration for H100 (LLaMA 8B)
```bash
export CRIU_CUDA_STREAMS=8           # H100 supports 8+ concurrent streams
export CRIU_CUDA_CHUNK_MB=512        # Larger chunks for high bandwidth
export CRIU_CUDA_USE_PINNED_MEM=1    # Essential for peak performance
export CUDA_INTERCEPT_RESTORE=1      # Enable parallel path
```

---

## Build & Deployment Commands

### Quick Build
```bash
cd /root/criu/plugins/cuda

# Build both libraries
make all

# Install
sudo make install
```

### Full CRIU Rebuild (if needed)
```bash
cd /root/criu
make clean
make -j$(nproc)
sudo make install
```

### Verify Installation
```bash
# Check plugin exists
ls -lh /usr/lib/criu/cuda_plugin.so

# Check interception library
ls -lh /usr/local/lib/libcuda_intercept.so

# Verify symbols
nm -D /usr/local/lib/libcuda_intercept.so | grep cudaMemcpy
```

---

## Troubleshooting Guide

### Issue: Context pool initialization fails
**Symptom**: "Failed to initialize GPU context pool" in logs
**Solution**:
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA runtime
ldconfig -p | grep cuda

# Test CUDA directly
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Interception library not loaded
**Symptom**: No "cuda_intercept:" messages in logs
**Solution**:
```bash
# Verify LD_PRELOAD is set
echo $LD_PRELOAD

# Check library exists and is readable
file /usr/local/lib/libcuda_intercept.so

# Test loading manually
LD_PRELOAD=/usr/local/lib/libcuda_intercept.so echo "Test"
```

### Issue: cudaMemcpy not intercepted
**Symptom**: Restore works but no speedup
**Solution**:
```bash
# Verify CUDA_INTERCEPT_RESTORE is set
echo $CUDA_INTERCEPT_RESTORE

# Check cuda-checkpoint uses cudaMemcpy (not cuMemcpy)
strace -e trace=write cuda-checkpoint --action restore --pid $PID 2>&1 | grep -i memcpy

# Enable debug output
export CUDA_INTERCEPT_DEBUG=1
```

### Issue: Restore crashes with segmentation fault
**Symptom**: Crash during GPU memory transfer
**Solution**:
```bash
# Disable parallel restore temporarily
export CRIU_CUDA_PARALLEL_RESTORE=0

# If it works, gradually increase streams
export CRIU_CUDA_STREAMS=2
export CRIU_CUDA_STREAMS=4

# Check for memory corruption
dmesg | grep -i "cuda\|gpu\|memory"
```

---

## Performance Monitoring

### Timing Breakdown Script
```bash
#!/bin/bash
# File: /root/gpu-load/measure_restore_times.sh

echo "=== CRIU Restore Performance Analysis ==="
echo "Date: $(date)"
echo

# Measure with timestamps
start=$(date +%s.%N)
criu restore --images-dir "$1" --shell-job -v4 2>&1 | tee restore.log
end=$(date +%s.%N)

total=$(echo "$end - $start" | bc)
echo
echo "=== Timing Breakdown ==="
echo "Total restore time: ${total}s"

# Extract GPU times
gpu_context=$(grep "Context creation" restore.log | awk '{print $NF}')
gpu_memory=$(grep "GPU memory transfer" restore.log | awk '{print $NF}')
cpu_memory=$(grep "CPU memory restore" restore.log | awk '{print $NF}')

echo "  GPU context: ${gpu_context}s"
echo "  GPU memory:  ${gpu_memory}s"
echo "  CPU memory:  ${cpu_memory}s"

# Calculate percentages
if [ -n "$gpu_memory" ]; then
    gpu_pct=$(echo "scale=1; $gpu_memory / $total * 100" | bc)
    echo "  GPU percentage: ${gpu_pct}%"
fi

if [ -n "$cpu_memory" ]; then
    cpu_pct=$(echo "scale=1; $cpu_memory / $total * 100" | bc)
    echo "  CPU percentage: ${cpu_pct}%"
fi
```

### Usage
```bash
chmod +x /root/gpu-load/measure_restore_times.sh
./measure_restore_times.sh /checkpoint/dir
```

---

## Risk Mitigation

### Phase 1 Risks (Context Pooling)
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU memory leak | Low | Medium | Add cleanup in fini(), monitor with nvidia-smi |
| Context conflicts | Low | High | Use cudaSetDevice() properly, test multi-GPU |
| Init failure | Low | Low | Non-fatal, fallback to on-demand creation |

### Phase 2 Risks (LD_PRELOAD)
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Break normal apps | Medium | High | Only intercept during CUDA_INTERCEPT_RESTORE=1 |
| Race conditions | Medium | High | Use thread-local storage, proper locking |
| Symbol conflicts | Low | Medium | Use RTLD_NEXT, test with real workloads |
| Memory corruption | Low | Critical | Extensive testing, bounds checking |

### Testing Checklist

#### Before Production Deployment
- [ ] Test with single GPU
- [ ] Test with multi-GPU (if applicable)
- [ ] Test with small checkpoint (<1 GB)
- [ ] Test with large checkpoint (>50 GB)
- [ ] Test normal application (not checkpoint) with LD_PRELOAD
- [ ] Verify no memory leaks (run 10+ restore cycles)
- [ ] Check GPU memory usage (nvidia-smi)
- [ ] Validate restored application functionality
- [ ] Measure performance improvement vs baseline
- [ ] Test error paths (kill during restore, corrupt checkpoint)

---

## Success Metrics

### Performance Targets
```
Phase 1 Complete:
  ✅ Context creation: 3.1s → 0.2s
  ✅ Total restore: 38.8s → 35.9s (8% improvement)

Phase 2 Complete:
  ✅ GPU memory: 7.9s → 1.8s
  ✅ Total restore: 35.9s → 30.0s (23% improvement)

Phase 3 Complete:
  ✅ CPU memory: 28s → 5s
  ✅ Total restore: 30s → 7s (5.5× improvement)
  🎯 TARGET ACHIEVED
```

### Quality Metrics
- ✅ Zero crashes in 100 consecutive restores
- ✅ No GPU memory leaks after 24 hours
- ✅ Restored application passes all functional tests
- ✅ Performance variance <5% between runs

---

## Timeline & Milestones

### Week 1: Phase 1 Implementation
- Day 1-2: Implement context pooling
- Day 3: Testing and validation
- Day 4: Performance benchmarking
- Day 5: Documentation and code review

### Week 2-3: Phase 2 Implementation
- Week 2 Day 1-3: Build LD_PRELOAD library
- Week 2 Day 4-5: Integration with plugin
- Week 3 Day 1-2: Testing and debugging
- Week 3 Day 3-5: Performance tuning

### Week 4-6: Phase 3 Implementation
- Week 4-5: CPU memory parallelization (see separate doc)
- Week 6: Integration testing and optimization

### Week 7: Production Deployment
- Final testing
- Documentation
- Monitoring setup
- Production rollout

---

## Contact & Support

### Questions?
- Refer to main analysis: `/root/gpu-load/CUDA_CHECKPOINT_ANALYSIS.md`
- CPU optimization details: `/root/gpu-load/CRIU_RESTORE_IMPROVEMENTS_IMPLEMENTATION.md`
- Code location: `/root/criu/plugins/cuda/`

### Key Files
```
/root/criu/plugins/cuda/
├── cuda_plugin.c                    # Main CRIU plugin
├── cuda_parallel_restore.c          # Parallel restore infrastructure
├── cuda_parallel_restore.h          # Header
├── cuda_intercept.c                 # NEW: LD_PRELOAD library
├── Makefile                         # Build system
└── README.md                        # Plugin documentation

/root/gpu-load/
├── CUDA_CHECKPOINT_ANALYSIS.md      # This document
├── NEXT_STEPS_CUDA_OPTIMIZATION.md  # Implementation guide
└── CRIU_RESTORE_IMPROVEMENTS_IMPLEMENTATION.md  # CPU optimizations
```

---

**Ready to start?** → Begin with Phase 1 (Context Pooling) for immediate results!
