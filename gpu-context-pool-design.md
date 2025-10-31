# GPU Context Pool Design for CRIU

## Problem Statement

Current restore creates GPU contexts from scratch:
- CUDA initialization: ~0.5-0.8s
- Context creation: ~0.2-0.3s  
- cuBLAS/library init: ~0.3-0.5s
- **Total overhead: ~1.0-1.5s per restore**

Goal: Pre-create contexts and reuse them to eliminate this overhead.

## Approach 1: NVIDIA MPS (Multi-Process Service)

### How It Works
```
┌─────────────────────────────────┐
│ nvidia-cuda-mps-server          │
│  - Single GPU context           │
│  - Shared by all processes      │
│  - Time-sliced execution        │
└────────────┬────────────────────┘
             │ Shared context
      ┌──────┴──────┬──────────┐
      │             │          │
   Process A    Process B   Process C
   (restored)   (restored)  (restored)
```

### Implementation
```bash
# 1. Start MPS before any CRIU operations
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# 2. All restored processes automatically use shared context
# No code changes needed!

# 3. Verify
echo "get_server_list" | nvidia-cuda-mps-control
```

### Pros
✅ Zero code changes needed
✅ Built into NVIDIA drivers
✅ Transparent to applications
✅ Reduces context creation to ~0.1s

### Cons
❌ Requires MPS daemon running
❌ All processes share one context (less isolation)
❌ Some performance overhead for concurrent execution
❌ May not work with all CUDA features

### Expected Improvement
5.7s → 5.1-5.3s (0.4-0.6s saved)

---

## Approach 2: CRIU Pre-Fork with CUDA Context

### How It Works
```
                    ┌─────────────────────────┐
                    │ criu-gpu-pool daemon    │
                    │  - Pre-initialized CUDA │
                    │  - cuDevicePrimaryCtx   │
                    │  - Ready to fork        │
                    └──────────┬──────────────┘
                               │ fork()
            ┌──────────────────┼──────────────────┐
            │                  │                  │
       ┌────▼────┐        ┌───▼────┐        ┌───▼────┐
       │ Restore │        │ Restore│        │ Restore│
       │ vLLM #1 │        │ vLLM #2│        │ vLLM #3│
       │ (inherits        │ (inherits       │ (inherits
       │  context)        │  context)       │  context)
       └─────────┘        └────────┘        └────────┘
```

### Key Insight
CUDA contexts CAN be inherited via fork() if done carefully:
```c
// In daemon process (criu-gpu-pool)
cuInit(0);
CUdevice device;
cuDeviceGet(&device, 0);
CUcontext ctx;
cuDevicePrimaryCtxRetain(&ctx, device);

// Now fork() - child inherits CUDA state!
pid_t child = fork();
if (child == 0) {
    // Child process already has CUDA initialized
    // CRIU can restore into this process
}
```

### Implementation Steps
1. Create `criu-gpu-pool` daemon
2. Daemon pre-initializes CUDA contexts
3. When CRIU needs to restore:
   - Daemon forks a child process
   - Child process already has CUDA ready
   - CRIU restores application state into child
   - Minimal CUDA re-initialization needed

### Pros
✅ Eliminates most context creation overhead
✅ Each restore gets isolated context (via fork)
✅ Works with standard CRIU workflow
✅ No MPS dependencies

### Cons
❌ CUDA + fork() can be tricky (not officially supported)
❌ Need daemon management
❌ Requires careful CUDA state handling
⚠️ May have issues with CUDA >= 12.0

### Expected Improvement
5.7s → 5.0-5.2s (0.5-0.7s saved)

---

## Approach 3: Persistent Helper Process with CUDA IPC

### How It Works
```
┌──────────────────────────────────────────┐
│ criu-gpu-helper (persistent daemon)      │
│  - Pre-allocated GPU memory pool         │
│  - Pre-created CUDA contexts             │
│  - Exports IPC handles                   │
└──────────────┬───────────────────────────┘
               │ IPC handles
        ┌──────┴─────┬──────────────┐
        │            │              │
   ┌────▼──────┐ ┌──▼────────┐ ┌──▼────────┐
   │ Restored  │ │ Restored  │ │ Restored  │
   │ Process A │ │ Process B │ │ Process C │
   │ (imports  │ │ (imports  │ │ (imports  │
   │  via IPC) │ │  via IPC) │ │  via IPC) │
   └───────────┘ └───────────┘ └───────────┘
```

### Implementation
```c
// In helper daemon
cudaIpcMemHandle_t ipc_handle;
cudaIpcGetMemHandle(&ipc_handle, device_ptr);
// Send handle to restored process via socket

// In restored process
cudaIpcOpenMemHandle(&local_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
// Now has access to pre-allocated GPU memory!
```

### Pros
✅ Official CUDA API (supported)
✅ Clean separation of concerns
✅ Can pre-allocate GPU memory pool
✅ Each process still gets own context

### Cons
❌ Only shares memory, not contexts (limited benefit)
❌ Complex IPC communication needed
❌ Still pays context creation cost (~0.5s)
❌ Memory must be pre-allocated to specific size

### Expected Improvement
5.7s → 5.5s (0.2s saved) - **MINIMAL BENEFIT**

---

## Approach 4: Modify cuda-checkpoint to Support Context Reuse

### How It Works
```c
// Modified cuda-checkpoint behavior
CUcontext get_or_create_context(int gpu_id) {
    static CUcontext cached_contexts[MAX_GPUS] = {0};
    
    if (cached_contexts[gpu_id] == 0) {
        // First time - create new context
        cuDevicePrimaryCtxRetain(&cached_contexts[gpu_id], gpu_id);
    }
    
    return cached_contexts[gpu_id]; // Reuse on subsequent restores
}
```

### Pros
✅ Most direct solution
✅ Optimal performance
✅ Works within existing architecture

### Cons
❌ Requires modifying NVIDIA's closed-source cuda-checkpoint
❌ Not feasible without NVIDIA cooperation

---

## Approach 5: CRIU Plugin Pre-Initialization (RECOMMENDED)

### How It Works
Keep context warm between restores by NEVER destroying it:

```c
// In cuda_plugin.c

static CUcontext g_warm_context = NULL;
static CUdevice g_warm_device = 0;
static bool g_context_pool_enabled = false;

int cuda_plugin_init(int stage) {
    if (stage == CR_PLUGIN_STAGE__RESTORE) {
        const char *pool_env = getenv("CRIU_CUDA_CONTEXT_POOL");
        if (pool_env && atoi(pool_env) == 1) {
            // Initialize CUDA context once during CRIU startup
            if (g_warm_context == NULL) {
                cuInit(0);
                cuDeviceGet(&g_warm_device, 0);
                cuDevicePrimaryCtxRetain(&g_warm_context, g_warm_device);
                cuCtxSetCurrent(g_warm_context);
                
                pr_info("GPU context pool initialized: ctx=%p\n", g_warm_context);
                g_context_pool_enabled = true;
            }
        }
        // ... rest of init
    }
}

int resume_device(int pid, int checkpointed, cuda_task_state_t initial_task_state) {
    // When restoring, CUDA context already exists in CRIU plugin process
    // cuda-checkpoint will create minimal additional state
    
    if (g_context_pool_enabled) {
        // Set context as current before calling cuda-checkpoint
        cuCtxSetCurrent(g_warm_context);
    }
    
    // Standard restore...
}
```

### Key Insight
The CRIU plugin process itself can maintain warm CUDA state!

### Architecture
```
┌──────────────────────────────────────────────┐
│ CRIU Process (criu restore ...)             │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │ cuda_plugin.so                         │ │
│  │  - g_warm_context (persistent)         │ │
│  │  - Initialized once                    │ │
│  │  - Reused across all restores          │ │
│  └────────────┬───────────────────────────┘ │
│               │ Warm context available      │
│               ▼                              │
│  ┌────────────────────────────────────────┐ │
│  │ cuda-checkpoint restore                │ │
│  │  - Finds CUDA already initialized      │ │
│  │  - Skips most init overhead            │ │
│  │  - Only restores application-specific  │ │
│  │    state                                │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### Pros
✅ **Simplest to implement** - just modify cuda_plugin.c
✅ No daemon needed
✅ Works within existing CRIU architecture
✅ Context lives in CRIU process itself
✅ Automatic cleanup when CRIU exits
✅ Can test immediately

### Cons
⚠️ Need to verify cuda-checkpoint respects existing context
⚠️ May conflict if cuda-checkpoint expects clean slate
⚠️ Limited to single restore at a time (sequential)

### Expected Improvement
5.7s → 5.0-5.3s (0.4-0.7s saved)

---

## Recommended Implementation: Hybrid Approach

Combine Approach 5 (CRIU plugin pre-init) with Approach 1 (MPS) as fallback:

```bash
# 1. Enable MPS for shared context (immediate, no code)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
nvidia-cuda-mps-control -d

# 2. Add context pool to CRIU plugin (1-2 days coding)
export CRIU_CUDA_CONTEXT_POOL=1
criu restore ...

# 3. Benchmark both approaches
./benchmark-context-pool.py
```

### Implementation Timeline

**Day 1: MPS Testing (Zero Code)**
- Start MPS daemon
- Test with current CRIU restore
- Measure performance impact
- Expected: 5.7s → 5.3s (0.4s saved)

**Day 2-3: CRIU Plugin Context Pool**
- Modify cuda_plugin.c to maintain warm context
- Add environment variable: CRIU_CUDA_CONTEXT_POOL
- Test with vLLM restore
- Expected: 5.7s → 5.0s (0.7s saved)

**Day 4: Combined Testing**
- MPS + warm context
- Measure cumulative benefit
- Expected: 5.7s → 4.8s (0.9s saved)

**Day 5: Production Hardening**
- Error handling
- Context cleanup
- Documentation

---

## Next Steps

1. **Immediate test (5 minutes):** Enable MPS and measure impact
2. **Quick win (2-3 days):** Implement CRIU plugin context pool
3. **Combined (1 week):** Both approaches for maximum benefit

Expected total improvement: **5.7s → 4.8s (16% faster)**

When combined with LD_PRELOAD parallel restore: **5.7s → 3.5s (38% faster!)**
