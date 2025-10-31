# Complete State Snapshot - GPU Checkpoint Optimization Project

**Date:** 2025-10-29 04:40 UTC
**Project:** vLLM CRIU GPU Checkpoint/Restore Optimization
**Current Performance:** 5.7s restore time (10.4x faster than 59s baseline)
**Goal:** Achieve sub-3s restore through parallelization and context pooling
**Status:** Ready to implement Option 2 (CRIU Plugin Context Pool)

---

## Executive Summary

### What We've Achieved So Far

1. ✅ **Optimized vLLM Configuration**
   - GPU memory utilization: 0.30 (down from 0.50)
   - Enabled --enforce-eager flag
   - Using safetensors format
   - Local checkpoint storage (not RAM-based)
   - **Result:** 59.2s → 6.3s → 5.7s (with persistence mode)

2. ✅ **TDD Investigation of CRIU Optimizations**
   - Tested readahead optimization in pagemap.c
   - Result: No improvement (I/O not the bottleneck)
   - Validated that GPU operations dominate, not disk I/O
   - **Lesson:** Don't optimize I/O, focus on GPU

3. ✅ **Deep Research Completed**
   - PhoenixOS GPU context pooling architecture (via subagent)
   - LD_PRELOAD interception for parallel GPU transfers (via subagent)
   - cuda-checkpoint binary analysis (via subagent)
   - **Discovery:** CRIU already has cuda_parallel_restore infrastructure built!

4. ✅ **Context Pool Design**
   - Analyzed 5 different approaches
   - Identified optimal solution: CRIU plugin persistent context
   - Created implementation code
   - **Ready to implement!**

### Current Performance Breakdown

```
Total Restore Time: 5.7s

Breakdown:
├── CRIU memory restore: 3.9s (68%)
├── GPU context creation: 1.0s (18%) ← TARGET for context pool
├── Process state restore: 0.4s (7%)
└── Network/FD/container: 0.4s (7%)
```

### Optimization Targets Identified

| Optimization | Time Saved | Complexity | Status |
|--------------|------------|------------|--------|
| GPU Context Pool | 0.7s | Low | **READY TO IMPLEMENT** |
| LD_PRELOAD Parallel | 1.5s | Medium | Code ready, pending test |
| CRIU Parallel VMA | 2.0s | High | Future work |
| **TOTAL POTENTIAL** | **4.2s** | | **Sub-2s possible!** |

---

## System Configuration

### Hardware
- **GPU:** NVIDIA (verified working)
- **Driver:** 570.158.01
- **CUDA:** 12.x
- **System:** Linux 6.11.0-29-generic

### Software Versions
- **CRIU:** Custom built v4.0-217-gae971c9fe at `/root/criu/criu/criu`
- **System CRIU:** v4.1.1 at `/usr/sbin/criu` (backed up to `/usr/sbin/criu.backup`)
- **vLLM:** docker.io/vllm/vllm-openai:latest
- **Model:** Qwen/Qwen2-1.5B-Instruct

### Current CRIU CUDA Plugin Status

```bash
Location: /root/criu/plugins/cuda/

Files:
- cuda_plugin.so (49KB) - Compiled and active ✅
- cuda_parallel_restore.c - Multi-stream infrastructure ✅
- cuda_parallel_restore.h - Headers ✅
- cuda_plugin.c (659 lines) - Main plugin code

Key Functions:
- cuda_parallel_restore_init() ✅ Compiled
- cuda_parallel_restore_memory() ✅ Compiled
- cuda_parallel_restore_buffer() ✅ Compiled
- cuda_parallel_restore_fini() ✅ Compiled

Status: Infrastructure exists but returns -ENOTSUP (not integrated)
```

### Environment Variables in Use

```bash
# vLLM Configuration
MODEL_ID=Qwen/Qwen2-1.5B-Instruct
GPU_MEM=0.30
MAX_MODEL_LEN=1024
CONT_NAME=vllm-llm-demo
API_PORT=8000

# CRIU Configuration (not yet set, but ready)
CRIU_CUDA_PARALLEL_RESTORE=1    # Enable parallel restore
CRIU_CUDA_STREAMS=8             # Number of CUDA streams
CRIU_CUDA_CHUNK_MB=256          # Chunk size for parallel
CRIU_CUDA_CONTEXT_POOL=1        # Enable context pool (NEW!)
```

### Container Configuration

```bash
Container: vllm-llm-demo
Image: docker.io/vllm/vllm-openai:latest
Runtime: Podman with CRIU support

Volumes:
- /opt/nvidia-libs:/opt/nvidia-libs:ro
- /models:/root/.cache/huggingface

Devices:
- /dev/nvidia0
- /dev/nvidiactl
- /dev/nvidia-uvm

Current Checkpoint:
Location: /var/lib/containers/storage/overlay-containers/<ID>/userdata/checkpoint
Size: 8.89 GB
Creation time: 15-16s
```

---

## Research Findings

### Finding 1: CUDA Parallel Restore Already Built

**Location:** `/root/criu/plugins/cuda/cuda_parallel_restore.c`

**What it does:**
- Splits GPU memory transfers into chunks
- Uses multiple CUDA streams (configurable 1-32)
- Round-robin distribution across streams
- Pinned memory for 2-3x bandwidth improvement
- Expected speedup: 60-70% faster GPU restore

**Why it's not active:**
- Returns `-ENOTSUP` at line 324
- cuda-checkpoint handles GPU internally
- Need LD_PRELOAD interception to activate

**Symbols verified:**
```
cuda_parallel_config_from_env  ✅
cuda_parallel_restore_buffer   ✅
cuda_parallel_restore_fini     ✅
cuda_parallel_restore_init     ✅
cuda_parallel_restore_memory   ✅
```

### Finding 2: PhoenixOS Context Pooling Strategy

**Paper:** "PhoenixOS: Concurrent OS-level GPU Checkpoint and Restore" (SOSP'25)

**Key Innovation:**
- Pre-allocate GPU contexts in daemon process
- Eliminates 3.1s context creation overhead
- Llama2-13B: 9.8s → 2.3s migration (4.3x faster)

**Architecture:**
```
┌─────────────────────┐
│ PhoenixOS Daemon    │
│ (phosd)             │
│  - Pre-created CUDA │
│  - Pre-created cuBLAS│
│  - Survives restarts│
└──────────┬──────────┘
           │ Context pool
    ┌──────┴──────┐
    │             │
Process A     Process B
(restored)    (restored)
```

**Our Adaptation:**
- Simpler: No daemon needed
- Keep context in CRIU plugin itself
- Expected: 5.7s → 5.0s (0.7s saved)

### Finding 3: LD_PRELOAD Interception Path

**Problem:** cuda-checkpoint uses single-threaded cudaMemcpy

**Solution:** Intercept and redirect to cuda_parallel_restore

**Implementation:**
```c
// libcuda_intercept.so
cudaError_t cudaMemcpy(...) {
    if (HostToDevice && size >= threshold) {
        // Use existing parallel infrastructure!
        return cuda_parallel_restore_buffer(...);
    }
    return original_cudaMemcpy(...);
}
```

**Expected:** 7.9s GPU → 1.8s GPU (4.4x faster for large transfers)

### Finding 4: CPU Memory Bottleneck

**Current:** Sequential preadv in CRIU restorer

**Opportunity:** Multi-threaded VMA restore

**Expected:** 3.9s → 0.8s (5x faster)

**Effort:** 3-4 weeks of CRIU C development

---

## Files Created During Research

### Documentation
1. `/root/gpu-load/CRIU_TDD_FINDINGS.md` - TDD investigation results
2. `/root/gpu-load/BENCHMARK_RESULTS_2025-10-29.md` - Readahead test results
3. `/root/gpu-load/CUDA_OPTIMIZATION_SYNTHESIS.md` - Overall strategy
4. `/root/gpu-load/gpu-context-pool-design.md` - 5 approaches analyzed
5. `/root/gpu-load/QUICK_START_CONTEXT_POOL.md` - Implementation guide
6. `/root/gpu-load/CURRENT_STATE_2025-10-29.md` - **THIS FILE**

### Implementation Code
1. `/root/gpu-load/criu-context-pool-implementation.c` - Ready to integrate
2. `/root/gpu-load/benchmark-criu-comparison.py` - Test script
3. `/root/gpu-load/benchmark-custom-criu.py` - Original test script

### Backups
1. `/root/criu/criu/pagemap.c` - Modified with readahead (lines 820-837)
2. `/usr/sbin/criu.backup` - Original system CRIU v4.1.1

---

## Next Step: Implement Option 2 (Context Pool)

### What We're Implementing

**Approach 5 from design document:** CRIU Plugin Pre-Initialization

**Key Insight:** The CRIU plugin process is persistent during checkpoint/restore operations. We can maintain a warm CUDA context in the plugin itself, eliminating repeated initialization overhead.

**Architecture:**
```
┌──────────────────────────────────────────┐
│ CRIU Process (criu restore ...)         │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ cuda_plugin.so                     │ │
│  │                                    │ │
│  │ static CUcontext g_warm_context;   │ │
│  │                                    │ │
│  │ First restore: cuInit() → 1.0s     │ │
│  │ Later restores: reuse → 0.05s ✅   │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### Changes Required

**File:** `/root/criu/plugins/cuda/cuda_plugin.c` (659 lines)

**Three modifications:**

1. **Add new code at top** (after includes, before line 21)
   - Add global variables for context pool
   - Add init_gpu_context_pool() function
   - Add fini_gpu_context_pool() function
   - Add prewarm_context_for_restore() function
   - **~150 lines of new code**

2. **Modify cuda_plugin_init()** (line 572)
   - Add context pool initialization
   - **~10 lines added**

3. **Modify resume_device()** (line 475)
   - Add prewarm call before restore
   - **~3 lines added**

4. **Modify cuda_plugin_fini()** (line 633)
   - Add context pool cleanup
   - **~5 lines added**

**Total changes:** ~170 lines added, 0 lines removed

### Expected Build Time
- Clean build: ~2 minutes
- Incremental: ~30 seconds

### Expected Test Time
- Single restore test: 10 seconds
- Full benchmark (3 iterations × 2 configs): ~5 minutes

---

## Testing Strategy

### Phase 1: Verify Build (5 minutes)

```bash
cd /root/criu/plugins/cuda

# Backup current version
cp cuda_plugin.c cuda_plugin.c.before-context-pool

# Apply changes (exact patch provided separately)
# ... edit cuda_plugin.c ...

# Rebuild
cd /root/criu
make clean && make

# Verify plugin rebuilt
ls -lh /root/criu/plugins/cuda/cuda_plugin.so
nm /root/criu/plugins/cuda/cuda_plugin.so | grep -i warm
```

### Phase 2: Test Context Pool Initialization (2 minutes)

```bash
# Enable context pool
export CRIU_CUDA_CONTEXT_POOL=1

# Test restore (check logs)
podman container restore --keep vllm-llm-demo 2>&1 | tee restore-test.log

# Verify context pool initialized
grep -i "context pool" restore-test.log
# Expected: "GPU context pool initialized successfully"
```

### Phase 3: Benchmark Performance (5 minutes)

```bash
# Create simple benchmark script
cat > /root/gpu-load/test-context-pool.sh << 'EOF'
#!/bin/bash
for i in {1..3}; do
    echo "=== Test $i ==="
    podman stop vllm-llm-demo 2>/dev/null || true
    sleep 1
    time podman container restore --keep vllm-llm-demo
    sleep 2
done
EOF

chmod +x /root/gpu-load/test-context-pool.sh

# Test WITHOUT context pool
unset CRIU_CUDA_CONTEXT_POOL
./test-context-pool.sh | tee results-without-pool.log

# Test WITH context pool
export CRIU_CUDA_CONTEXT_POOL=1
./test-context-pool.sh | tee results-with-pool.log

# Compare
echo "Without pool:" && grep "real" results-without-pool.log
echo "With pool:" && grep "real" results-with-pool.log
```

### Phase 4: Full Benchmark (10 minutes)

```bash
# Run comprehensive comparison
./benchmark-criu-comparison.py

# Expected output:
# System CRIU (no pool):     5.7s
# System CRIU (with pool):   5.0s
# Improvement: 0.7s (12% faster) ✅
```

---

## Success Criteria

### Must Have (Minimum Viable)
- ✅ Code compiles without errors
- ✅ CRIU restore still works (no regression)
- ✅ Context pool initializes successfully
- ✅ Can enable/disable via CRIU_CUDA_CONTEXT_POOL

### Should Have (Expected Performance)
- ✅ 0.3-0.7s improvement in restore time
- ✅ Context reused across multiple restores
- ✅ No memory leaks (verified with multiple restores)

### Nice to Have (Stretch Goals)
- ✅ Detailed logging of context pool lifecycle
- ✅ Graceful fallback if pool init fails
- ✅ Documentation in code comments

---

## Rollback Plan

If anything goes wrong:

```bash
# Restore original cuda_plugin.c
cd /root/criu/plugins/cuda
cp cuda_plugin.c.before-context-pool cuda_plugin.c

# Rebuild
cd /root/criu
make clean && make

# Verify original behavior restored
podman container restore --keep vllm-llm-demo

# Total rollback time: 3 minutes
```

---

## Known Risks & Mitigations

### Risk 1: Context Pool Init Fails
**Impact:** Restore may fail completely
**Probability:** Low (CUDA is stable)
**Mitigation:** Graceful fallback - continue without pool
**Code:** Check return value, log warning, disable pool

### Risk 2: cuda-checkpoint Conflicts with Existing Context
**Impact:** Restore corruption or crash
**Probability:** Medium-Low (need to test)
**Mitigation:** Extensive testing, verify cuda-checkpoint behavior
**Rollback:** Disable CRIU_CUDA_CONTEXT_POOL=0

### Risk 3: Memory Leak from Unreleased Context
**Impact:** GPU memory slowly fills up
**Probability:** Low (we call release in fini)
**Mitigation:** Monitor GPU memory, test multiple restores
**Detection:** nvidia-smi after 10+ restores

### Risk 4: Performance Doesn't Improve
**Impact:** Wasted development time
**Probability:** Medium (depends on cuda-checkpoint behavior)
**Mitigation:** Quick validation test before full integration
**Pivot:** Move to LD_PRELOAD approach instead

---

## Timeline

### Today (2-3 hours)
- ✅ Create state snapshot (this file)
- ✅ Create exact patch for cuda_plugin.c
- ⏳ Apply patch
- ⏳ Build and test
- ⏳ Initial benchmark

### Tomorrow (2-3 hours)
- ⏳ Full benchmark suite
- ⏳ Multiple restore testing (verify no leaks)
- ⏳ Documentation updates
- ⏳ Commit to git

### Day 3 (1-2 hours)
- ⏳ Production validation
- ⏳ Performance tuning (if needed)
- ⏳ Next steps planning (LD_PRELOAD?)

**Total: 2-3 days to completion**

---

## Next Actions (IMMEDIATE)

1. **Review exact patch** - See EXACT_CHANGES_CUDA_PLUGIN.patch
2. **Apply changes** - Edit cuda_plugin.c
3. **Build** - make clean && make
4. **Test** - Quick restore test
5. **Benchmark** - Full performance comparison

**READY TO PROCEED!** All research complete, code ready, plan validated.

---

## Contact Points / References

- PhoenixOS Paper: https://arxiv.org/abs/2405.12079
- PhoenixOS Repo: https://github.com/SJTU-IPADS/PhoenixOS
- NVIDIA CUDA Checkpoint API: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html
- CRIU CUDA Plugin: /root/criu/plugins/cuda/
- Implementation Reference: /root/gpu-load/criu-context-pool-implementation.c

---

**Last Updated:** 2025-10-29 04:40 UTC
**Status:** READY FOR IMPLEMENTATION
**Next File:** See EXACT_CHANGES_CUDA_PLUGIN.patch for precise modifications
