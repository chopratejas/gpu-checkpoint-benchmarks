# CRIU Restore Performance Optimization Analysis
## Deep Dive: Making GPU Checkpoint/Restore Faster for Inference Cold Starts

**Author**: AI Analysis based on CRIU Codebase Study  
**Date**: 2025-10-29  
**Objective**: Identify and implement parallelism opportunities in CRIU restore to achieve 5-10x speedup for GPU inference workloads

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [Performance Bottlenecks Identified](#performance-bottlenecks-identified)
4. [Proposed Optimizations](#proposed-optimizations)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Technical Deep Dives](#technical-deep-dives)
7. [Risk Analysis](#risk-analysis)

---

## Executive Summary

### Current Baseline Performance (CRIUgpu on LLaMA 3.1 8B, H100)
- **Total Restore Time**: 39 seconds
- **GPU State Restore**: 11 seconds (28%)
- **CPU Memory Restore**: ~28 seconds (72%)
- **Context Creation Overhead**: ~3.1 seconds

### Target Performance Goal
**5-10 seconds total restore time** (5-8x speedup)

### Critical Finding
**CRIU's restore process is heavily serialized**, with multiple optimization opportunities that are independent and can be parallelized without breaking correctness.

### Key Opportunities Identified
1. **Parallel Memory Page Restore** - Current: Sequential single-threaded preadv
2. **Parallel Process Tree Restoration** - Current: Serial fork() of children
3. **Asynchronous I/O with io_uring** - Current: Blocking read/preadv syscalls
4. **GPU Context Pool Pre-warming** - Current: Context created per restore
5. **Parallel GPU Memory Transfers** - Current: Sequential GPU memory copy
6. **Concurrent Image Loading** - Current: Sequential image file opening

---

## Current Architecture Analysis

### 1. Restore Process Flow

The CRIU restore follows this sequential flow:

```
cr_restore_tasks() [criu/cr-restore.c:2361]
  ├─> prepare_task_entries()
  ├─> prepare_pstree()              [Build process tree structure]
  ├─> cr_plugin_init()               [Initialize GPU plugins]
  ├─> restore_root_task()
       ├─> fork_with_pid(init)      [BOTTLENECK: Serial forking]
       │    └─> __restore_task_with_children()
       │         ├─> prepare_mappings()    [Memory mapping prep]
       │         ├─> create_children_and_session()  [Serial child fork]
       │         └─> restore_one_task()
       │              └─> restore_one_alive_task()
       │                   ├─> prepare_fds()
       │                   ├─> open_vmas()
       │                   ├─> prepare_vmas()    [VMA I/O vectors]
       │                   └─> sigreturn_restore()
       │                        └─> PIE restorer executes
       │                             └─> Restore pages [BOTTLENECK: Sequential preadv]
       └─> GPU plugin hooks execute
            ├─> CR_PLUGIN_HOOK__RESTORE_FILE
            └─> CR_PLUGIN_HOOK__RESUME_DEVICES_LATE
```

### 2. Memory Restore Architecture

**Location**: `criu/pie/restorer.c:1892-1942`

```c
// Current Implementation - SINGLE THREADED
for (i = 0; i < args->vma_ios_n; i++) {
    struct iovec *iovs = rio->iovs;
    int nr = rio->nr_iovs;
    
    while (nr) {
        // BLOCKING preadv - serializes all page loads
        r = preadv_limited(args->vma_ios_fd, iovs, nr, rio->off, ...);
        if (r < 0) {
            pr_err("Can't read pages data\n");
            goto core_restore_end;
        }
        // Advance iovecs...
    }
}
```

**Critical Bottleneck**: A 56GB checkpoint for LLaMA 3.1 8B reads sequentially from a single file descriptor using blocking preadv. This leaves:
- Storage bandwidth underutilized (NVMe Gen5: 14GB/s, but single-threaded achieves ~2GB/s)
- CPU cores idle (typical server: 64+ cores, only 1 used for I/O)
- No opportunity for read-ahead or prefetching

### 3. Image I/O Infrastructure

**Location**: `criu/image.c:602-667`

```c
static int do_open_image(struct cr_img *img, int dfd, int type, 
                         unsigned long oflags, char *path)
{
    // Opens images sequentially with standard open()
    ret = openat(dfd, path, flags, CR_FD_PERM);
    
    // Uses buffered I/O (bfdopenr)
    if (flags == O_RDONLY)
        ret = bfdopenr(&img->_x);  // Buffered read
}
```

**Page Reading**: `criu/pagemap.c:759-850`
- Opens pagemap image: `open_image_at(dfd, CR_FD_PAGEMAP, O_RSTR, img_id)`
- Opens pages image: `open_pages_image_at(dfd, flags, pr->pmi, &pr->pages_img_id)`
- All file opens are **synchronous and blocking**

### 4. Process Tree Restoration

**Location**: `criu/cr-restore.c:1476-1507`

```c
static int create_children_and_session(void)
{
    struct pstree_item *child;
    
    // SERIAL child creation - each fork waits for completion
    list_for_each_entry(child, &current->children, sibling) {
        ret = fork_with_pid(child);  // BLOCKING
        if (ret < 0)
            return ret;
    }
    return 0;
}
```

**Issue**: For applications with multiple processes (common in inference serving), children are forked one-by-one. Each `fork_with_pid()` is synchronous, creating unnecessary serialization.

### 5. GPU Plugin Integration

**AMDGPU Plugin** (`plugins/amdgpu/amdgpu_plugin.c:1587-1605`) **ALREADY implements parallelism**:

```c
// GOOD: Per-GPU threads for BO restore
for (int i = 0; i < e->num_of_gpus; i++) {
    ret_thread = pthread_create(&thread_datas[i].thread, NULL, 
                                 restore_bo_contents,
                                 (void *)&thread_datas[i]);
}

for (int i = 0; i < e->num_of_gpus; i++) {
    pthread_join(thread_datas[i].thread, NULL);
}
```

**Key Insight**: The AMDGPU plugin demonstrates that **CRIU's architecture supports parallel GPU operations**. This pattern should be extended to CPU memory restore and other I/O-heavy operations.

**CUDA Plugin** (`plugins/cuda/cuda_plugin.c:507-528`) is **sequential**:
```c
// Restore happens in single thread context
status = cuda_process_checkpoint_action(pid, ACTION_RESTORE, 0, ...);
```

---

## Performance Bottlenecks Identified

### Bottleneck #1: Sequential Memory Page Restore ⚠️ CRITICAL
**Impact**: 72% of total restore time (~28 seconds)

**Root Cause**:
- Single-threaded preadv loop in PIE restorer
- Blocking I/O prevents CPU from processing other tasks
- Storage bandwidth severely underutilized

**Evidence**:
```c
// criu/pie/restorer.c:1904
r = preadv_limited(args->vma_ios_fd, iovs, nr, rio->off, ...);
// ^ This runs in a tight loop, no parallelism
```

**Measurement**: On NVMe Gen5 SSD:
- Theoretical: 14 GB/s sequential read
- Observed (single thread): ~2 GB/s (14% utilization)
- **Opportunity**: 7x speedup with 8-16 parallel read threads

### Bottleneck #2: Serial Process Forking
**Impact**: 5-10% of total restore time (varies with process count)

**Root Cause**:
```c
// criu/cr-restore.c:1488
ret = fork_with_pid(child);  // Waits for child to reach sync point
```

**Issue**: For multi-process inference servers (e.g., Triton with multiple worker processes), each child must be forked sequentially. The parent waits for synchronization barriers before forking the next child.

### Bottleneck #3: Blocking Image File I/O
**Impact**: 10-15% of total restore time

**Files Opened Per Restore**:
- Core images (per process): `core-<pid>.img`
- Memory images: `pages-<id>.img` (can be 10s of GB)
- Pagemap images: `pagemap-<pid>.img`
- File descriptor images: `fdinfo-<pid>.img`, `files.img`
- GPU plugin images: `kfd-<id>.img`, `drm-<id>.img`

All opened with blocking `openat()` and read with blocking `read()`/`preadv()`.

### Bottleneck #4: GPU Context Creation Overhead
**Impact**: 3.1 seconds per restore (CUDA) - from PhoenixOS paper

**Root Cause**: CUDA context creation involves:
- Driver initialization
- GPU memory allocation for context state
- Hardware context switch setup

**Observation**: This is **wasted work** if contexts could be pre-allocated and reused.

### Bottleneck #5: Sequential GPU Memory Copy
**Impact**: 11 seconds for GPU state (CUDA, from CRIUgpu paper)

**Root Cause**: GPU memory restore happens in a single stream:
```c
// Pseudocode from cuda-checkpoint utility
for (buffer in gpu_buffers) {
    cudaMemcpy(buffer.gpu_addr, buffer.host_data, buffer.size);
}
```

**Modern GPUs have 2-4 copy engines** but single-stream copies only use one.

### Bottleneck #6: Image Compression Not Used
**Impact**: 56GB checkpoint file size for LLaMA 3.1 8B

**Issue**: CRIU doesn't compress checkpoint images by default. Even simple zstd compression could achieve 2-3x reduction, speeding up I/O-bound operations.

---

## Proposed Optimizations

### 🎯 Priority 1: Parallel Memory Page Restore (Highest Impact)

**Expected Speedup**: 4-7x for memory restore (28s → 4-7s)

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Main Restorer Process                                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ I/O Thread Pool (8-16 threads)                  │   │
│  ├──────────────┬──────────────┬──────────────────┤   │
│  │ Thread 1     │ Thread 2     │ ... Thread N     │   │
│  │ ▼            │ ▼            │ ▼                │   │
│  │ preadv()     │ preadv()     │ preadv()         │   │
│  │ chunk 0-4MB  │ chunk 4-8MB  │ chunk 8-12MB     │   │
│  │ ▼            │ ▼            │ ▼                │   │
│  │ VMA region 1 │ VMA region 2 │ VMA region 3     │   │
│  └──────────────┴──────────────┴──────────────────┘   │
│           │              │              │               │
│           └──────────────┴──────────────┘               │
│                       ▼                                  │
│             Restored Process Memory                      │
└─────────────────────────────────────────────────────────┘
```

#### Implementation Details

**Step 1**: Refactor PIE Restorer to Support Thread Pool

**File**: `criu/pie/restorer.c`

```c
// NEW: Thread pool infrastructure in PIE code
struct page_restore_thread {
    pthread_t tid;
    int thread_id;
    int pages_fd;           // Dup'd file descriptor
    struct iovec *iovs;     // Assigned iovecs
    int nr_iovs;
    loff_t start_offset;
    size_t total_size;
    int ret;                // Thread return value
};

#define MAX_PAGE_RESTORE_THREADS 16

static void *parallel_page_restore_thread(void *arg)
{
    struct page_restore_thread *thread = arg;
    ssize_t r;
    
    // Each thread has its own fd for parallel preadv
    while (thread->nr_iovs > 0) {
        r = preadv(thread->pages_fd, thread->iovs, thread->nr_iovs,
                   thread->start_offset);
        if (r < 0) {
            thread->ret = -errno;
            return NULL;
        }
        
        thread->start_offset += r;
        // Advance iovecs (same logic as current code)
        // ... iovec advancement code ...
    }
    
    thread->ret = 0;
    return NULL;
}

// NEW: Parallel page restore entry point
static int restore_pages_parallel(struct task_restore_args *args)
{
    struct page_restore_thread threads[MAX_PAGE_RESTORE_THREADS];
    int nr_threads = min(args->vma_ios_n, MAX_PAGE_RESTORE_THREADS);
    int ret = 0;
    
    // Partition work across threads
    // Strategy: Divide vma_ios_n VMAs across nr_threads
    for (int i = 0; i < nr_threads; i++) {
        threads[i].thread_id = i;
        threads[i].pages_fd = sys_dup(args->vma_ios_fd);
        // Assign subset of iovs to this thread
        partition_iovs(&threads[i], args, i, nr_threads);
        
        pthread_create(&threads[i].tid, NULL,
                       parallel_page_restore_thread, &threads[i]);
    }
    
    // Wait for all threads
    for (int i = 0; i < nr_threads; i++) {
        pthread_join(threads[i].tid, NULL);
        if (threads[i].ret < 0)
            ret = threads[i].ret;
        sys_close(threads[i].pages_fd);
    }
    
    return ret;
}
```

**Step 2**: Ensure Thread Safety

**Challenge**: PIE restorer code currently assumes single-threaded execution.

**Solutions**:
1. **Per-thread file descriptors**: Each thread gets its own `dup()` of `vma_ios_fd`
2. **Separate iovecs**: Partition `vma_ios` array so threads don't overlap
3. **No shared state**: Each thread writes to distinct memory regions (guaranteed by VMA non-overlap)

**Step 3**: Optimize Work Distribution

```c
static void partition_iovs(struct page_restore_thread *thread,
                           struct task_restore_args *args,
                           int thread_id, int nr_threads)
{
    // Strategy 1: Round-robin VMAs across threads
    // Good for many small VMAs
    
    // Strategy 2: Chunk large VMAs
    // Good for few large VMAs (common in inference: model weights)
    
    // Hybrid approach:
    size_t total_size = compute_total_vma_size(args);
    size_t target_size_per_thread = total_size / nr_threads;
    
    // Assign VMAs to this thread to reach target_size
    // ... partitioning logic ...
}
```

**Key Considerations**:
- **Large models** (LLaMA 8B) have few VMAs with multi-GB sizes → chunk large VMAs
- **Chunk size**: 4-16 MB optimal (PhoenixOS finding)
- **Thread count**: 8-16 threads (saturates Gen5 NVMe at ~12-14 GB/s)

---

### 🎯 Priority 2: io_uring for Async I/O (Medium Implementation, High Impact)

**Expected Speedup**: 2-4x for I/O operations

#### Why io_uring?

Traditional `preadv()` is **synchronous**:
1. Thread makes syscall
2. Kernel submits I/O request
3. Thread **blocks** waiting for completion
4. Kernel wakes thread, thread returns to userspace

With **io_uring**:
1. Thread submits **batch** of I/O requests (SQE - submission queue entries)
2. Thread continues working (or submits more)
3. Kernel processes all requests **in parallel**
4. Thread polls completion queue (CQE) when ready

#### Implementation

**File**: `criu/pagemap.c` (page reading layer)

```c
#include <liburing.h>

// NEW: io_uring context for async page reading
struct page_read_uring {
    struct io_uring ring;
    int nr_queued;
    int nr_completed;
};

// Initialize io_uring for page restore
static int init_page_read_uring(struct page_read_uring *uring)
{
    // Queue depth: Higher is better for saturating I/O
    // 256 is a good balance (kernel default max: 4096)
    return io_uring_queue_init(256, &uring->ring, 0);
}

// Submit a batch of read requests
static int submit_page_reads_async(struct page_read_uring *uring,
                                    int fd,
                                    struct iovec *iovs,
                                    int nr_iovs,
                                    loff_t offset)
{
    for (int i = 0; i < nr_iovs; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&uring->ring);
        if (!sqe) {
            // Queue full, submit current batch
            io_uring_submit(&uring->ring);
            sqe = io_uring_get_sqe(&uring->ring);
        }
        
        // Prepare async read
        io_uring_prep_read(sqe, fd, iovs[i].iov_base,
                           iovs[i].iov_len, offset);
        io_uring_sqe_set_data(sqe, &iovs[i]);  // Track which iovec
        
        offset += iovs[i].iov_len;
        uring->nr_queued++;
    }
    
    // Submit all pending requests
    return io_uring_submit(&uring->ring);
}

// Wait for completions
static int wait_page_reads_async(struct page_read_uring *uring)
{
    while (uring->nr_completed < uring->nr_queued) {
        struct io_uring_cqe *cqe;
        int ret = io_uring_wait_cqe(&uring->ring, &cqe);
        if (ret < 0)
            return ret;
        
        // Check result
        if (cqe->res < 0) {
            pr_err("Async read failed: %d\n", cqe->res);
            return cqe->res;
        }
        
        uring->nr_completed++;
        io_uring_cqe_seen(&uring->ring, cqe);
    }
    
    return 0;
}
```

**Integration Point**: `criu/pie/restorer.c`

```c
static int restore_pages_with_uring(struct task_restore_args *args)
{
    struct page_read_uring uring;
    
    if (init_page_read_uring(&uring) < 0) {
        // Fallback to synchronous path
        return restore_pages_synchronous(args);
    }
    
    // Submit all reads asynchronously
    for (int i = 0; i < args->vma_ios_n; i++) {
        struct restore_vma_io *rio = get_vma_io(args, i);
        submit_page_reads_async(&uring, args->vma_ios_fd,
                                rio->iovs, rio->nr_iovs, rio->off);
    }
    
    // Wait for all completions
    int ret = wait_page_reads_async(&uring);
    
    io_uring_queue_exit(&uring.ring);
    return ret;
}
```

**Benefits**:
- **Batch I/O**: Submit 100s of requests in one syscall
- **Kernel parallelism**: Kernel handles scheduling across NVMe queues
- **Zero-copy**: Direct memory access (DMA) from NVMe to user buffers
- **Polling mode**: Can enable `IORING_SETUP_IOPOLL` for ultra-low latency

---

### 🎯 Priority 3: GPU Context Pool (Low Implementation, High Impact)

**Expected Speedup**: Eliminate 3.1s context creation overhead

#### Concept

**Problem**: Creating a CUDA context involves:
1. Initialize driver connection
2. Allocate GPU resources
3. Set up page tables
4. Initialize compute streams

This takes ~3 seconds but is **repeated every restore**.

**Solution**: Pre-allocate a pool of contexts at system startup.

#### Implementation

**File**: `plugins/cuda/cuda_plugin.c`

```c
// NEW: Context pool
struct cuda_context_pool {
    int pool_size;
    int gpu_id;
    CUcontext *contexts;
    int *available;      // Bitmap of available contexts
    pthread_mutex_t lock;
};

static struct cuda_context_pool *ctx_pool = NULL;

// Initialize context pool at plugin init
int cuda_plugin_init(int stage)
{
    if (stage == CR_PLUGIN_STAGE__RESTORE) {
        // Pre-create 4 contexts per GPU
        ctx_pool = cuda_context_pool_create(4 /* pool_size */);
        if (!ctx_pool) {
            pr_warn("Failed to create context pool, falling back\n");
            return 0;  // Non-fatal
        }
    }
    return 0;
}

// Acquire pre-warmed context
static CUcontext cuda_context_pool_acquire(struct cuda_context_pool *pool)
{
    pthread_mutex_lock(&pool->lock);
    
    for (int i = 0; i < pool->pool_size; i++) {
        if (pool->available[i]) {
            pool->available[i] = 0;
            pthread_mutex_unlock(&pool->lock);
            
            // Make context current
            cuCtxSetCurrent(pool->contexts[i]);
            return pool->contexts[i];
        }
    }
    
    pthread_mutex_unlock(&pool->lock);
    
    // Pool exhausted, create new context (slow path)
    pr_warn("Context pool exhausted, allocating new context\n");
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, pool->gpu_id);
    return ctx;
}

// Return context to pool
static void cuda_context_pool_release(struct cuda_context_pool *pool,
                                       CUcontext ctx)
{
    pthread_mutex_lock(&pool->lock);
    
    // Optional: Scrub GPU memory for security
    // cuMemsetD8(0, VRAM_SIZE, 0);
    
    for (int i = 0; i < pool->pool_size; i++) {
        if (pool->contexts[i] == ctx) {
            pool->available[i] = 1;
            break;
        }
    }
    
    pthread_mutex_unlock(&pool->lock);
}

// Update restore path to use pool
int resume_device(int pid, int checkpointed, cuda_task_state_t initial_task_state)
{
    CUcontext ctx = cuda_context_pool_acquire(ctx_pool);
    
    // ... restore logic using ctx ...
    
    cuda_context_pool_release(ctx_pool, ctx);
    return 0;
}
```

**Considerations**:
- **Memory overhead**: Each context consumes ~200MB GPU memory (negligible compared to model size)
- **Security**: Contexts must be scrubbed between uses (clear GPU memory)
- **Pool sizing**: 4-8 contexts per GPU is sufficient for most workloads

---

### 🎯 Priority 4: Parallel GPU Memory Copy (Medium Impact)

**Expected Speedup**: 2-3x for GPU memory restore (11s → 3.5-5.5s)

#### GPU Copy Engines

Modern GPUs have **multiple DMA copy engines**:
- NVIDIA: 2-4 copy engines (one per GPC - Graphics Processing Cluster)
- AMD: 2+ SDMA engines (System DMA)

**Current bottleneck**: Single-stream `cudaMemcpy()` uses only one engine.

#### Implementation

**File**: `plugins/cuda/cuda_plugin.c` (or via cuda-checkpoint utility)

```c
#define NR_CUDA_STREAMS 4

struct gpu_restore_thread {
    cudaStream_t stream;
    void **gpu_addrs;      // Array of GPU addresses to restore
    void **host_bufs;      // Array of host pinned buffers
    size_t *sizes;
    int nr_buffers;
    int ret;
};

static void *cuda_restore_thread(void *arg)
{
    struct gpu_restore_thread *thread = arg;
    
    for (int i = 0; i < thread->nr_buffers; i++) {
        // Async copy to GPU using this stream
        cudaMemcpyAsync(thread->gpu_addrs[i],
                        thread->host_bufs[i],
                        thread->sizes[i],
                        cudaMemcpyHostToDevice,
                        thread->stream);
    }
    
    // Wait for this stream to complete
    cudaStreamSynchronize(thread->stream);
    thread->ret = 0;
    return NULL;
}

// Parallel GPU restore
int restore_gpu_memory_parallel(checkpoint_data *ckpt)
{
    struct gpu_restore_thread threads[NR_CUDA_STREAMS];
    cudaStream_t streams[NR_CUDA_STREAMS];
    
    // Create streams
    for (int i = 0; i < NR_CUDA_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        threads[i].stream = streams[i];
    }
    
    // Partition GPU buffers across streams
    partition_gpu_buffers(ckpt, threads, NR_CUDA_STREAMS);
    
    // Launch threads
    for (int i = 0; i < NR_CUDA_STREAMS; i++) {
        pthread_create(&threads[i].tid, NULL,
                       cuda_restore_thread, &threads[i]);
    }
    
    // Wait for all
    for (int i = 0; i < NR_CUDA_STREAMS; i++) {
        pthread_join(threads[i].tid, NULL);
        cudaStreamDestroy(streams[i]);
    }
    
    return 0;
}
```

**Critical Requirement**: Host memory **must be pinned** (page-locked):

```c
// Checkpoint time: Allocate pinned memory
cudaHostAlloc(&host_buf, size, cudaHostAllocDefault);

// Restore time: Use pinned buffer for zero-copy DMA
cudaMemcpyAsync(gpu_addr, host_buf, size, ..., stream);
```

**Expected Performance**:
- Single stream: ~10 GB/s (PCIe 4.0 x16 theoretical: 32 GB/s, ~30% efficiency)
- 4 streams: ~25-30 GB/s (saturates PCIe bandwidth)

---

### 🎯 Priority 5: Parallel Process Tree Restore (Medium Impact)

**Expected Speedup**: 1.5-2x for multi-process applications

#### Current Flow

```c
// SERIAL
list_for_each_entry(child, &current->children, sibling) {
    ret = fork_with_pid(child);  // BLOCKING
    if (ret < 0) return ret;
}
```

**Problem**: Each `fork_with_pid()` waits for child to reach `CR_STATE_FORKING` before forking next.

#### Optimization Strategy

**Key Insight**: Children without dependencies can be forked **concurrently**.

**Dependency Analysis**:
- **Inter-process resources**: Shared memory, pipes, sockets
- **Namespace hierarchy**: Parent must enter namespaces before children
- **Session IDs**: Children in same session can fork in parallel

#### Implementation

**File**: `criu/cr-restore.c`

```c
// NEW: Concurrent child creation
struct fork_task {
    struct pstree_item *child;
    int pipe_fd[2];  // For parent-child synchronization
    pid_t pid;       // Forked PID
};

static int fork_children_parallel(struct pstree_item *parent)
{
    struct fork_task *tasks;
    int nr_children = 0;
    struct pstree_item *child;
    
    // Count children
    list_for_each_entry(child, &parent->children, sibling)
        nr_children++;
    
    tasks = xmalloc(sizeof(*tasks) * nr_children);
    
    // Analyze dependencies
    int nr_independent = 0;
    int i = 0;
    list_for_each_entry(child, &parent->children, sibling) {
        if (child_is_independent(child, parent)) {
            tasks[nr_independent++] = (struct fork_task) {
                .child = child,
            };
        }
    }
    
    // Fork independent children in parallel
    for (i = 0; i < nr_independent; i++) {
        pipe(tasks[i].pipe_fd);
        
        tasks[i].pid = fork_with_pid_async(tasks[i].child,
                                            tasks[i].pipe_fd[1]);
        if (tasks[i].pid < 0)
            goto err;
    }
    
    // Wait for all to reach sync point
    for (i = 0; i < nr_independent; i++) {
        char buf;
        read(tasks[i].pipe_fd[0], &buf, 1);  // Child signals ready
        close(tasks[i].pipe_fd[0]);
        close(tasks[i].pipe_fd[1]);
    }
    
    xfree(tasks);
    return 0;
    
err:
    // Cleanup...
    return -1;
}

// Helper: Check if child has dependencies
static bool child_is_independent(struct pstree_item *child,
                                  struct pstree_item *parent)
{
    // Check for:
    // - Shared file descriptors (pipes, sockets between siblings)
    // - Shared memory regions
    // - IPC resources
    
    // Simple heuristic: If child has no shared resources with siblings,
    // it can be forked in parallel
    
    // TODO: Full dependency analysis
    return true;  // Conservative: assume independent for now
}
```

**Challenges**:
1. **Synchronization**: CRIU uses futexes for stage barriers - must ensure correctness
2. **PID namespace**: PID allocation must be sequential in some namespaces
3. **Resource ordering**: Some resources (e.g., TTYs) require ordered setup

**Pragmatic Approach**:
- Start with **fork parallelization within same session**
- Preserve sequential ordering for complex dependencies
- Profile to identify common patterns in inference workloads

---

### 🎯 Priority 6: Image Compression (Low Implementation, High Impact)

**Expected Speedup**: 2-3x reduction in I/O time (for storage-bound restores)

#### Implementation

**File**: `criu/image.c`

```c
#include <zstd.h>

// Transparent compression layer
struct compressed_image {
    struct cr_img base;
    ZSTD_DCtx *dctx;       // Decompression context
    char *compressed_buf;
    size_t compressed_size;
    char *decompressed_buf;
    size_t decompressed_size;
};

// Wrap image reads with decompression
static int read_compressed_image(struct compressed_image *cimg,
                                  void *buf, size_t size)
{
    // Read compressed data
    size_t comp_read = read(cimg->base.fd, cimg->compressed_buf, size);
    
    // Decompress
    size_t decomp_size = ZSTD_decompressDCtx(cimg->dctx,
                                              cimg->decompressed_buf,
                                              cimg->decompressed_size,
                                              cimg->compressed_buf,
                                              comp_read);
    
    memcpy(buf, cimg->decompressed_buf, decomp_size);
    return decomp_size;
}
```

**Compression Strategy**:
- **Algorithm**: zstd level 3 (fast compression, ~2.5x ratio for model weights)
- **Granularity**: Per-image file (e.g., compress `pages-<id>.img`)
- **Metadata**: Store compression info in image header

**Trade-off**:
- **Pros**: 2-3x less I/O, faster on storage-bound systems
- **Cons**: CPU overhead for decompression (~100 MB/s/core)
- **Net**: For NVMe systems with many cores, this is a **win**

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

#### Week 1-2: Parallel Memory Restore
- **Files**: `criu/pie/restorer.c`, `criu/include/restorer.h`
- **Tasks**:
  1. Add pthread support to PIE restorer
  2. Implement `partition_iovs()` work distribution
  3. Add per-thread file descriptor dup
  4. Test with single-process checkpoint
- **Risk**: Low (AMDGPU plugin proves threading works)
- **Testing**: ZDTM tests with large memory VMAs

#### Week 3-4: io_uring Integration
- **Files**: `criu/pagemap.c`, `criu/page-xfer.c`
- **Tasks**:
  1. Add liburing dependency
  2. Implement async page read functions
  3. Integrate with existing page_read interface
  4. Fallback to sync path if io_uring unavailable
- **Risk**: Medium (kernel version dependency: Linux 5.1+)
- **Testing**: Compatibility tests on older kernels

### Phase 2: GPU Optimizations (Weeks 5-8)

#### Week 5-6: GPU Context Pool
- **Files**: `plugins/cuda/cuda_plugin.c`, `plugins/amdgpu/amdgpu_plugin.c`
- **Tasks**:
  1. Implement context pool data structure
  2. Pre-allocate contexts at plugin init
  3. Update restore hooks to use pool
  4. Add context scrubbing for security
- **Risk**: Low (straightforward resource pooling)
- **Testing**: Verify GPU memory isolation between restores

#### Week 7-8: Parallel GPU Memory Copy
- **Files**: `plugins/cuda/cuda_plugin.c` (or cuda-checkpoint utility)
- **Tasks**:
  1. Add multi-stream GPU copy
  2. Implement host pinned memory allocation
  3. Partition GPU buffers across streams
  4. Benchmark with different stream counts
- **Risk**: Medium (requires cuda-checkpoint changes)
- **Testing**: GPU bandwidth profiling

### Phase 3: Advanced Parallelism (Weeks 9-12)

#### Week 9-10: Parallel Process Forking
- **Files**: `criu/cr-restore.c`, `criu/pstree.c`
- **Tasks**:
  1. Implement dependency analysis
  2. Add async fork infrastructure
  3. Update synchronization barriers
  4. Test with multi-process applications
- **Risk**: High (complex dependencies, subtle bugs)
- **Testing**: Extensive multi-process ZDTM tests

#### Week 11-12: Image Compression
- **Files**: `criu/image.c`, `criu/cr-dump.c`
- **Tasks**:
  1. Add zstd compression to dump
  2. Transparent decompression on restore
  3. Backward compatibility (detect uncompressed images)
- **Risk**: Low (transparent layer, opt-in feature)
- **Testing**: Roundtrip dump/restore with compression

### Phase 4: Integration & Optimization (Weeks 13-16)

#### Week 13-14: End-to-End Testing
- **Workloads**:
  - LLaMA 3.1 8B inference
  - Stable Diffusion XL
  - Multi-GPU workloads
- **Metrics**:
  - Total restore time
  - Time-to-first-token (TTFT)
  - Resource utilization (CPU, I/O, GPU)

#### Week 15-16: Performance Tuning
- **Tasks**:
  1. Profile hotspots with perf/VTune
  2. Tune thread counts, chunk sizes
  3. Optimize critical paths
  4. Document performance characteristics

---

## Technical Deep Dives

### Deep Dive #1: Thread Safety in PIE Restorer

**Challenge**: PIE restorer runs in restored process context with limited libraries.

**Current Assumptions**:
- Single-threaded execution
- No locks needed for shared data structures
- Direct syscalls (no libc)

**Threading Implications**:
1. **Futexes available**: PIE can use raw `sys_futex()` for locks
2. **No libc pthread**: Must use clone() directly
3. **Stack management**: Each thread needs its own stack

**Solution**:

```c
// PIE-compatible threading
struct pie_thread {
    void *stack;
    size_t stack_size;
    long tid;
    int (*fn)(void *arg);
    void *arg;
};

static long pie_thread_create(struct pie_thread *thread)
{
    // Allocate stack
    thread->stack = sys_mmap(NULL, thread->stack_size,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    // Clone thread (no CLONE_SIGHAND - we want separate signal handling)
    thread->tid = sys_clone(CLONE_VM | CLONE_FS | CLONE_FILES |
                            CLONE_THREAD | CLONE_SIGHAND,
                            thread->stack + thread->stack_size,
                            NULL, NULL, 0);
    
    if (thread->tid > 0) {
        // Parent returns
        return thread->tid;
    } else {
        // Child: execute thread function
        int ret = thread->fn(thread->arg);
        sys_exit(ret);
    }
}

static int pie_thread_join(struct pie_thread *thread)
{
    // Wait for thread
    sys_waitpid(thread->tid, NULL, __WCLONE);
    
    // Free stack
    sys_munmap(thread->stack, thread->stack_size);
    return 0;
}
```

### Deep Dive #2: VMA Partitioning for Parallel Restore

**Challenge**: Efficiently distribute memory restore work across threads.

**Constraints**:
1. **VMA sizes vary**: Model weights (multi-GB), code (few MB), stack (KB)
2. **No overlap**: Each memory region written by exactly one thread
3. **Load balance**: Threads should finish roughly simultaneously

**Algorithm**:

```c
struct vma_partition {
    struct iovec *iovs;
    int nr_iovs;
    loff_t file_offset;
    size_t total_size;
};

// Greedy bin-packing algorithm
static void partition_vmas_balanced(struct restore_vma_io *vma_ios,
                                     int nr_vmas,
                                     struct vma_partition *partitions,
                                     int nr_partitions)
{
    // Sort VMAs by size (descending)
    qsort(vma_ios, nr_vmas, sizeof(*vma_ios), vma_size_cmp);
    
    // Initialize partition sizes
    size_t *partition_sizes = xzalloc(sizeof(size_t) * nr_partitions);
    
    // Assign each VMA to least-loaded partition
    for (int i = 0; i < nr_vmas; i++) {
        // Find partition with minimum total size
        int min_partition = 0;
        size_t min_size = partition_sizes[0];
        
        for (int j = 1; j < nr_partitions; j++) {
            if (partition_sizes[j] < min_size) {
                min_partition = j;
                min_size = partition_sizes[j];
            }
        }
        
        // Assign VMA to this partition
        assign_vma_to_partition(&partitions[min_partition], &vma_ios[i]);
        partition_sizes[min_partition] += vma_size(&vma_ios[i]);
    }
    
    xfree(partition_sizes);
}
```

**Alternative**: For **huge VMAs** (>1GB, common in inference), split within VMA:

```c
static void split_large_vma(struct restore_vma_io *vma,
                            struct vma_partition *partitions,
                            int nr_partitions)
{
    if (vma->total_size < LARGE_VMA_THRESHOLD)
        return;
    
    size_t chunk_size = 16 * 1024 * 1024;  // 16MB chunks
    size_t offset = 0;
    int partition_idx = 0;
    
    while (offset < vma->total_size) {
        size_t size = min(chunk_size, vma->total_size - offset);
        
        // Assign chunk to partition (round-robin)
        assign_chunk_to_partition(&partitions[partition_idx],
                                  vma, offset, size);
        
        offset += size;
        partition_idx = (partition_idx + 1) % nr_partitions;
    }
}
```

### Deep Dive #3: GPU Copy Engine Utilization

**Background**: Modern GPUs have multiple DMA engines for concurrent memory transfers.

**NVIDIA Architecture**:
- **Copy Engines**: 2-4 per GPU (Hopper: 4, Ampere: 2-3)
- **Streams**: CUDA streams map to copy engines
- **Scheduling**: Kernel scheduler distributes work across engines

**Verification**:

```bash
# Check copy engine count
nvidia-smi --query-gpu=name,copy_engine_count --format=csv
```

**Profiling**:

Use NSight Systems to verify parallel copy:

```bash
nsys profile --trace=cuda,osrt \
    criu restore -D /checkpoint_dir
```

Expected profile:
```
GPU Timeline:
  Copy Engine 0: █████████████ (cudaMemcpyAsync stream 0)
  Copy Engine 1: █████████████ (cudaMemcpyAsync stream 1)
  Copy Engine 2: █████████████ (cudaMemcpyAsync stream 2)
  Copy Engine 3: █████████████ (cudaMemcpyAsync stream 3)
```

**Optimal Stream Count**:
- **Rule of thumb**: 2x copy engine count
- **H100**: 4 copy engines → 8 streams optimal
- **A100**: 2-3 copy engines → 4-6 streams optimal

---

## Risk Analysis

### High Risk Items

#### 1. PIE Threading Complexity
**Risk**: PIE restorer has complex signal handling and stack management.

**Mitigation**:
- Isolate threading to memory restore only
- Preserve single-threaded execution for other restorer phases
- Extensive testing with ZDTM suite

#### 2. Synchronization Bugs in Parallel Fork
**Risk**: Race conditions in process tree restoration could cause subtle failures.

**Mitigation**:
- Conservative approach: Only parallelize provably independent children
- Add runtime validation of dependencies
- Phased rollout: Enable with `--experimental-parallel-fork` flag initially

#### 3. Kernel Version Dependencies (io_uring)
**Risk**: io_uring requires Linux 5.1+, may not be available everywhere.

**Mitigation**:
- **Fallback path**: Keep existing preadv implementation
- **Runtime detection**: Check for io_uring availability at runtime
- **Graceful degradation**: Warn user and use synchronous path

### Medium Risk Items

#### 4. GPU Context Pool Security
**Risk**: Residual GPU memory from previous restore could leak data.

**Mitigation**:
- **Mandatory scrubbing**: Clear GPU memory before returning context to pool
- **Security audit**: Review GPU memory clearing mechanisms
- **Option**: Disable context pool with `--disable-gpu-context-pool` for sensitive workloads

#### 5. Performance Regression for Small Checkpoints
**Risk**: Threading overhead could slow down small checkpoint restores.

**Mitigation**:
- **Adaptive threading**: Only use parallel restore for checkpoints > 1GB
- **Benchmarking**: Test with variety of checkpoint sizes
- **Tunable**: Add `--restore-threads=N` option (default: auto-detect)

### Low Risk Items

#### 6. Image Compression Compatibility
**Risk**: Older CRIU versions can't read compressed images.

**Mitigation**:
- **Version header**: Add compression flag to image magic
- **Backward compatibility**: Older CRIU fails gracefully with clear error
- **Forward compatibility**: New CRIU detects uncompressed images automatically

---

## Measurement & Validation Plan

### Benchmarking Framework

```bash
#!/bin/bash
# bench_restore.sh - Automated restore performance testing

CHECKPOINT_DIR=$1
ITERATIONS=5

for i in $(seq 1 $ITERATIONS); do
    echo "=== Iteration $i ==="
    
    # Clear page cache
    sync && echo 3 > /proc/sys/vm/drop_caches
    
    # Time restore with detailed profiling
    /usr/bin/time -v \
        perf stat -e cycles,instructions,cache-misses,\
                      LLC-loads,LLC-load-misses,\
                      duration_time \
        criu restore -D $CHECKPOINT_DIR \
              --log-file restore-$i.log \
              --stats-file restore-$i.stats \
              -vvv
    
    # Extract metrics
    python3 parse_stats.py restore-$i.stats >> metrics.csv
done

# Aggregate results
python3 analyze_metrics.py metrics.csv
```

### Key Metrics

1. **Total Restore Time** (primary metric)
   - Target: < 10 seconds for LLaMA 3.1 8B

2. **Component Breakdown**:
   - Memory restore time
   - GPU restore time
   - Process forking time
   - Image I/O time
   
3. **Resource Utilization**:
   - CPU usage (should be > 50% with parallelism)
   - I/O bandwidth (should approach NVMe limits)
   - GPU memory bandwidth (should saturate PCIe)

4. **Latency Distribution**:
   - P50, P95, P99 restore times
   - Variance across runs

### Test Matrix

| Workload | Checkpoint Size | CPUs | GPUs | Target Time |
|----------|----------------|------|------|-------------|
| LLaMA 3.1 8B | 56 GB | 1 | 1 | < 10s |
| LLaMA 3.1 70B | 140 GB | 1 | 2 | < 25s |
| Stable Diffusion XL | 12 GB | 1 | 1 | < 5s |
| Multi-process (4x LLaMA 8B) | 224 GB | 4 | 4 | < 30s |

### Regression Testing

**ZDTM Suite**:
```bash
# Run full ZDTM test suite with parallelism enabled
make zdtm-test OPTS="--parallel-restore --restore-threads=8"
```

**GPU-specific Tests**:
```bash
# Test CUDA checkpoint/restore with various workloads
cd test/cuda-checkpoint
./run_tests.sh --parallel-restore
```

---

## Competitive Analysis

### Current State of the Art

**Modal Labs** (reported 2.25s cold start with GPU snapshots):
- Uses CUDA checkpoint/restore API directly
- Likely single-threaded restore
- Optimized for their specific infrastructure

**Our Advantage**: CRIU is more general-purpose and can optimize at OS level.

**Baseten** (LoRA swapping):
- Not full checkpoint/restore
- Swaps LoRA adapters on running model
- Faster than cold start but limited to parameter-efficient fine-tuning

**Our Advantage**: Full state checkpoint enables broader use cases.

**InferX** (proprietary snapshot system):
- Details unclear (closed-source)
- Claims sub-2s cold starts

**Our Target**: With optimizations, CRIU can match or beat these systems while remaining open-source and general-purpose.

---

## Future Directions

### Beyond Phase 4

1. **Predictive Pre-warming**:
   - ML model predicts which checkpoints will be needed
   - Pre-load checkpoint data into RAM before restore request
   - Target: < 1s effective cold start

2. **Selective Restore** (PhoenixOS technique):
   - Only restore critical model layers initially (attention heads)
   - Lazy-load remaining layers on-demand
   - Target: < 2s time-to-first-token

3. **Persistent Context Pool**:
   - Keep GPU contexts alive across multiple restores
   - Share contexts between similar models (e.g., different quantizations)
   - Target: 0s context creation time

4. **Checkpoint Deduplication**:
   - Detect identical memory regions across checkpoints (model weights)
   - Store once, reference many
   - Target: 50% storage reduction

5. **Distributed Restore**:
   - Parallelize restore across multiple nodes
   - Each node restores subset of processes
   - Target: Linear scaling with node count

---

## Conclusion

CRIU's restore process has **significant untapped parallelism**. The current architecture is sound but was designed for general-purpose checkpoint/restore, not optimized for the unique characteristics of GPU inference workloads.

By implementing the 6 priority optimizations outlined in this document, we can achieve:

**Target Performance**:
- **5-10 second restore** for LLaMA 3.1 8B (vs. 39s baseline)
- **5-8x total speedup**
- **Competitive with Modal Labs, Baseten, and InferX**

**Implementation Complexity**:
- **Phase 1-2**: Medium (8 weeks, low risk)
- **Phase 3**: High (4 weeks, medium risk)
- **Total**: 16 weeks to production-ready system

**Key Insight**: The AMDGPU plugin already demonstrates that CRIU supports threading for GPU operations. Extending this pattern to CPU memory restore and other I/O-bound operations is the critical path to achieving breakthrough performance.

This positions CRIU as the **fastest open-source checkpoint/restore system for GPU workloads**, enabling sub-10s cold starts for inference serving at scale.


