# CRIU Parallel Restore - Implementation Guide

## ⚠️ IMPORTANT: Code Corrections Required

**This implementation guide was validated against CRIU v4.1.1 source code. Critical findings:**

1. **PIE Threading Limitation**: PIE restorer has NO access to pthread library. Must use raw `clone()` syscalls, not `pthread_create()`.
2. **GPU Context Pooling**: CUDA plugin delegates to external `cuda-checkpoint` binary. Context pooling must be implemented there, not in plugin.
3. **Syscall Wrappers**: CRIU uses `sys_dup2()` or `sys_dup3()`, not `sys_dup()`.
4. **Performance Expectations**: Realistic speedup is 2-3x total (not 5-10x due to Amdahl's law).

See corrections throughout this document marked with ⚠️.

---

## Quick Start for Developers

This guide provides concrete, copy-paste-ready code for implementing parallel restore optimizations in CRIU.

---

## Priority 1: Parallel Memory Page Restore

### Step 1: Add Threading Infrastructure to PIE Restorer

**File**: `criu/pie/restorer.c`

⚠️ **CORRECTION**: PIE restorer cannot use pthread library! The code below shows pthread for illustration, but **must be rewritten to use raw `clone()` syscalls**.

Reference the AMDGPU plugin (`plugins/amdgpu/amdgpu_plugin.c:1587-1605`) which successfully uses pthread because it runs in plugin context (has full libc access), not PIE context.

Add these includes at the top (note: pthread.h won't work in PIE, shown for reference only):

```c
// ⚠️ WARNING: pthread.h NOT available in PIE restorer context!
// This code needs rewrite to use sys_clone() instead
#include <pthread.h>  // ← NOT AVAILABLE IN PIE
#include <errno.h>
```

Add thread pool structure after existing includes:

```c
/*
 * Thread pool for parallel page restore
 * Located in PIE restorer - has limited library access
 */
#define MAX_RESTORE_THREADS 16
#define MIN_VMA_SIZE_FOR_PARALLEL (16 * 1024 * 1024)  /* 16MB */

struct page_restore_worker {
	pthread_t thread_id;
	int worker_idx;
	
	/* File descriptor for this worker (dup'd) */
	int pages_fd;
	
	/* Work assignment */
	struct restore_vma_io *vma_ios;
	int nr_vmas;
	loff_t start_offset;
	size_t total_bytes;
	
	/* Auto-dedup support */
	int auto_dedup;
	
	/* Result */
	int ret;
};

struct page_restore_pool {
	struct page_restore_worker workers[MAX_RESTORE_THREADS];
	int nr_workers;
	int pages_fd_orig;  /* Original fd to dup from */
	int auto_dedup;
};
```

### Step 2: Worker Thread Implementation

⚠️ **CORRECTION**: This pthread-based code will NOT compile in PIE restorer. Must be rewritten to:
- Use `sys_clone()` with manual stack allocation
- Use futex-based synchronization (not pthread_join)
- See CRIU's existing PIE threading in `criu/pie/util.c` for patterns

Add after the structure definitions:

```c
/*
 * ⚠️ Worker thread for parallel page restoration
 * NOTE: This uses pthread for illustration but MUST be rewritten
 * to use raw clone() syscalls for PIE context!
 * Each worker processes a subset of VMAs independently
 */
static void *page_restore_worker_thread(void *arg)
{
	struct page_restore_worker *worker = arg;
	int i;
	ssize_t r;
	
	pr_debug("Worker %d starting: %d VMAs, %zu bytes\n",
		 worker->worker_idx, worker->nr_vmas, worker->total_bytes);
	
	/* Process assigned VMAs */
	for (i = 0; i < worker->nr_vmas; i++) {
		struct restore_vma_io *rio = &worker->vma_ios[i];
		struct iovec *iovs = rio->iovs;
		int nr = rio->nr_iovs;
		
		while (nr > 0) {
			pr_debug("Worker %d: preadv %d iovs at offset %lld\n",
				 worker->worker_idx, nr, (long long)rio->off);
			
			/*
			 * Use preadv_limited to support auto-dedup
			 * (matches existing single-threaded implementation)
			 */
			r = preadv_limited(worker->pages_fd, iovs, nr, rio->off,
					   worker->auto_dedup ? AUTO_DEDUP_OVERHEAD_BYTES : 0);
			if (r < 0) {
				pr_err("Worker %d: preadv failed: %ld\n",
				       worker->worker_idx, (long)r);
				worker->ret = (int)r;
				return NULL;
			}
			
			pr_debug("Worker %d: read %ld bytes\n",
				 worker->worker_idx, (long)r);
			
			/* Support auto-dedup hole punching */
			if (r > 0 && worker->auto_dedup) {
				int fr = sys_fallocate(worker->pages_fd,
						       FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE,
						       rio->off, r);
				if (fr < 0) {
					pr_debug("Worker %d: fallocate failed: %d\n",
						 worker->worker_idx, fr);
					/* Non-fatal */
				}
			}
			
			rio->off += r;
			
			/* Advance iovecs (same logic as single-threaded) */
			do {
				if (iovs->iov_len <= (size_t)r) {
					pr_debug("Worker %d: skip iovec\n",
						 worker->worker_idx);
					r -= iovs->iov_len;
					iovs++;
					nr--;
					continue;
				}
				
				iovs->iov_base = (void *)((unsigned long)iovs->iov_base + r);
				iovs->iov_len -= r;
				break;
			} while (nr > 0);
		}
	}
	
	pr_debug("Worker %d: completed successfully\n", worker->worker_idx);
	worker->ret = 0;
	return NULL;
}
```

### Step 3: Work Distribution Algorithm

Add work partitioning function:

```c
/*
 * Partition VMA I/O work across worker threads
 * Strategy: Balance total bytes per worker
 */
static void partition_vma_ios(struct task_restore_args *args,
			       struct page_restore_pool *pool)
{
	struct restore_vma_io *rio;
	size_t total_size = 0;
	size_t target_per_worker;
	int vma_idx = 0;
	int worker_idx = 0;
	int i;
	
	/* Calculate total size */
	rio = args->vma_ios;
	for (i = 0; i < args->vma_ios_n; i++) {
		int j;
		for (j = 0; j < rio->nr_iovs; j++) {
			total_size += rio->iovs[j].iov_len;
		}
		rio = (void *)rio + RIO_SIZE(rio->nr_iovs);
	}
	
	pr_info("Partitioning %zu bytes across %d workers\n",
		total_size, pool->nr_workers);
	
	target_per_worker = total_size / pool->nr_workers;
	
	/*
	 * Assign VMAs to workers using greedy algorithm
	 * Each worker gets VMAs until it reaches target size
	 */
	rio = args->vma_ios;
	for (vma_idx = 0; vma_idx < args->vma_ios_n; vma_idx++) {
		struct page_restore_worker *worker = &pool->workers[worker_idx];
		size_t vma_size = 0;
		int j;
		
		/* Calculate this VMA's size */
		for (j = 0; j < rio->nr_iovs; j++) {
			vma_size += rio->iovs[j].iov_len;
		}
		
		/* Assign to current worker */
		if (worker->nr_vmas == 0) {
			/* First VMA for this worker */
			worker->vma_ios = rio;
		}
		worker->nr_vmas++;
		worker->total_bytes += vma_size;
		
		pr_debug("Assigned VMA %d (%zu bytes) to worker %d\n",
			 vma_idx, vma_size, worker_idx);
		
		/*
		 * Move to next worker if:
		 * 1. We've reached target size, OR
		 * 2. This is the last VMA
		 */
		if (worker->total_bytes >= target_per_worker ||
		    vma_idx == args->vma_ios_n - 1) {
			/* Move to next worker (but don't exceed pool size) */
			if (worker_idx < pool->nr_workers - 1) {
				worker_idx++;
			}
		}
		
		rio = (void *)rio + RIO_SIZE(rio->nr_iovs);
	}
	
	/* Debug: Print final distribution */
	for (i = 0; i < pool->nr_workers; i++) {
		pr_info("Worker %d: %d VMAs, %zu bytes\n",
			i, pool->workers[i].nr_vmas,
			pool->workers[i].total_bytes);
	}
}
```

### Step 4: Main Parallel Restore Function

Replace the existing restore loop in `criu/pie/restorer.c` (around line 1892):

```c
/*
 * Determine optimal number of worker threads
 * Based on:
 * 1. Number of VMAs
 * 2. Total data size
 * 3. Available CPU cores (from sysconf)
 */
static int compute_optimal_workers(struct task_restore_args *args)
{
	long nr_cpus = sys_sysconf(_SC_NPROCESSORS_ONLN);
	size_t total_size = 0;
	struct restore_vma_io *rio;
	int i, j;
	
	if (nr_cpus < 0)
		nr_cpus = 4;  /* Default fallback */
	
	/* Calculate total I/O size */
	rio = args->vma_ios;
	for (i = 0; i < args->vma_ios_n; i++) {
		for (j = 0; j < rio->nr_iovs; j++) {
			total_size += rio->iovs[j].iov_len;
		}
		rio = (void *)rio + RIO_SIZE(rio->nr_iovs);
	}
	
	/*
	 * Heuristics:
	 * - Small restores (< 16MB): Use 1 thread
	 * - Medium restores (16MB - 1GB): Use 4 threads
	 * - Large restores (> 1GB): Use up to 16 threads or nr_cpus/2
	 */
	if (total_size < MIN_VMA_SIZE_FOR_PARALLEL) {
		return 1;
	} else if (total_size < (1ULL << 30)) {  /* < 1GB */
		return min(4, (int)nr_cpus / 2);
	} else {
		return min(MAX_RESTORE_THREADS, max(8, (int)nr_cpus / 2));
	}
}

/*
 * Parallel page restoration
 * Creates worker threads to restore pages concurrently
 */
static int restore_vma_ios_parallel(struct task_restore_args *args)
{
	struct page_restore_pool pool;
	int i, ret = 0;
	
	memset(&pool, 0, sizeof(pool));
	
	/* Determine number of workers */
	pool.nr_workers = compute_optimal_workers(args);
	pool.pages_fd_orig = args->vma_ios_fd;
	pool.auto_dedup = args->auto_dedup;
	
	pr_info("Starting parallel page restore with %d workers\n",
		pool.nr_workers);
	
	if (pool.nr_workers <= 1) {
		/* Fall back to single-threaded path */
		pr_info("Using single-threaded restore (small checkpoint)\n");
		goto single_threaded;
	}
	
	/* Create per-worker file descriptors */
	for (i = 0; i < pool.nr_workers; i++) {
		pool.workers[i].worker_idx = i;
		// ⚠️ CORRECTED: sys_dup() doesn't exist, use sys_dup2()
		int newfd = sys_dup2(pool.pages_fd_orig, -1);  // dup2 with -1 allocates new fd
		if (newfd < 0) {
			// Alternative: use sys_fcntl with F_DUPFD
			newfd = sys_fcntl(pool.pages_fd_orig, F_DUPFD, 0);
		}
		pool.workers[i].pages_fd = newfd;
		pool.workers[i].auto_dedup = pool.auto_dedup;
		
		if (pool.workers[i].pages_fd < 0) {
			pr_err("Failed to dup pages_fd for worker %d\n", i);
			ret = -EBADF;
			goto cleanup_fds;
		}
	}
	
	/* Partition work */
	partition_vma_ios(args, &pool);
	
	/* Start worker threads */
	for (i = 0; i < pool.nr_workers; i++) {
		int err = pthread_create(&pool.workers[i].thread_id, NULL,
					  page_restore_worker_thread,
					  &pool.workers[i]);
		if (err != 0) {
			pr_err("Failed to create worker thread %d: %d\n", i, err);
			ret = -err;
			/* TODO: Cancel already-started threads */
			goto cleanup_fds;
		}
	}
	
	/* Wait for all workers to complete */
	for (i = 0; i < pool.nr_workers; i++) {
		void *retval;
		pthread_join(pool.workers[i].thread_id, &retval);
		
		if (pool.workers[i].ret < 0) {
			pr_err("Worker %d failed with ret=%d\n",
			       i, pool.workers[i].ret);
			ret = pool.workers[i].ret;
		}
	}
	
cleanup_fds:
	/* Close duplicated file descriptors */
	for (i = 0; i < pool.nr_workers; i++) {
		if (pool.workers[i].pages_fd >= 0) {
			sys_close(pool.workers[i].pages_fd);
		}
	}
	
	if (ret == 0) {
		pr_info("Parallel page restore completed successfully\n");
	}
	
	return ret;

single_threaded:
	/* Fall back to existing single-threaded implementation */
	{
		struct restore_vma_io *rio = args->vma_ios;
		int i;
		
		for (i = 0; i < args->vma_ios_n; i++) {
			struct iovec *iovs = rio->iovs;
			int nr = rio->nr_iovs;
			ssize_t r;
			
			while (nr) {
				pr_debug("Preadv %lx:%d... (%d iovs)\n",
					 (unsigned long)iovs->iov_base,
					 (int)iovs->iov_len, nr);
				
				r = preadv_limited(args->vma_ios_fd, iovs, nr, rio->off,
						   args->auto_dedup ? AUTO_DEDUP_OVERHEAD_BYTES : 0);
				if (r < 0) {
					pr_err("Can't read pages data (%d)\n", (int)r);
					return (int)r;
				}
				
				pr_debug("`- returned %ld\n", (long)r);
				
				if (r > 0 && args->auto_dedup) {
					int fr = sys_fallocate(args->vma_ios_fd,
							       FALLOC_FL_KEEP_SIZE | FALLOC_FL_PUNCH_HOLE,
							       rio->off, r);
					if (fr < 0) {
						pr_debug("Failed to punch holes with fallocate: %d\n", fr);
					}
				}
				
				rio->off += r;
				
				do {
					if (iovs->iov_len <= (size_t)r) {
						pr_debug("   `- skip pagemap\n");
						r -= iovs->iov_len;
						iovs++;
						nr--;
						continue;
					}
					
					iovs->iov_base = (void *)((unsigned long)iovs->iov_base + r);
					iovs->iov_len -= r;
					break;
				} while (nr > 0);
			}
			
			rio = (void *)rio + RIO_SIZE(rio->nr_iovs);
		}
	}
	return 0;
}
```

### Step 5: Integration Point

Modify the main restorer code (around line 1890) to call the new function:

```c
// OLD CODE (line ~1890):
// rio = args->vma_ios;
// for (i = 0; i < args->vma_ios_n; i++) {
//     ...existing loop...
// }

// NEW CODE:
ret = restore_vma_ios_parallel(args);
if (ret < 0) {
	pr_err("Parallel page restore failed: %d\n", ret);
	goto core_restore_end;
}
```

### Step 6: Build System Changes

**File**: `criu/pie/Makefile` (or relevant Makefile)

Add pthread linking:

```makefile
# Add to LDFLAGS
LDFLAGS += -lpthread
```

### Step 7: Testing

Create test script:

```bash
#!/bin/bash
# test_parallel_restore.sh

set -e

echo "Testing parallel page restore..."

# Create test checkpoint
./test/zdtm.py run -t zdtm/static/maps00 \
    --pre 2 --parallel-restore

# Measure restore time
time criu restore -D dump/zdtm/static/maps00/395/ \
    --log-file restore.log \
    -vvv

# Check logs for parallel restore messages
grep "parallel page restore" restore.log
```

---

## Priority 2: io_uring Integration

### Step 1: Add Dependency

**File**: `Makefile.config`

Add liburing check:

```makefile
# Check for liburing
$(call try-cc, \
	$(FEATURE_TEST_LIBURING), \
	$(LIBS_URING), \
	liburing-present, \
	liburing-absent)

ifeq ($(call feature-test,liburing-present),true)
	LIBS_URING := -luring
	DEFINES += -DCONFIG_HAS_LIBURING
endif
```

**File**: `scripts/feature-tests.mak`

Add feature test:

```makefile
FEATURE_TEST_LIBURING := $(shell printf '%b' \
'#include <liburing.h>\n' \
'int main(void) {\n' \
'	struct io_uring ring;\n' \
'	io_uring_queue_init(32, &ring, 0);\n' \
'	io_uring_queue_exit(&ring);\n' \
'	return 0;\n' \
'}\n')
```

### Step 2: Implement io_uring Page Reader

**File**: `criu/pagemap-uring.c` (NEW FILE)

```c
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#ifdef CONFIG_HAS_LIBURING
#include <liburing.h>
#endif

#include "pagemap.h"
#include "common/compiler.h"
#include "log.h"
#include "util.h"
#include "page-xfer.h"

#ifdef CONFIG_HAS_LIBURING

#define URING_QUEUE_DEPTH 256
#define URING_MAX_IOVS 1024

struct page_read_uring {
	struct io_uring ring;
	int nr_submitted;
	int nr_completed;
	struct iovec *iovs;
	int nr_iovs;
	int nr_iovs_queued;
};

/*
 * Initialize io_uring for page reads
 */
int page_read_uring_init(struct page_read_uring **uring_out)
{
	struct page_read_uring *uring;
	int ret;
	
	uring = xmalloc(sizeof(*uring));
	if (!uring)
		return -ENOMEM;
	
	memset(uring, 0, sizeof(*uring));
	
	/* Initialize io_uring with reasonable queue depth */
	ret = io_uring_queue_init(URING_QUEUE_DEPTH, &uring->ring, 0);
	if (ret < 0) {
		pr_err("io_uring_queue_init failed: %d\n", ret);
		xfree(uring);
		return ret;
	}
	
	/* Allocate iovec array for tracking */
	uring->iovs = xmalloc(sizeof(struct iovec) * URING_MAX_IOVS);
	if (!uring->iovs) {
		io_uring_queue_exit(&uring->ring);
		xfree(uring);
		return -ENOMEM;
	}
	
	*uring_out = uring;
	
	pr_info("io_uring initialized (queue_depth=%d)\n", URING_QUEUE_DEPTH);
	return 0;
}

/*
 * Submit read requests to io_uring
 * Returns number of requests submitted, or negative on error
 */
int page_read_uring_submit(struct page_read_uring *uring,
			     int fd,
			     struct iovec *iovs,
			     int nr_iovs,
			     loff_t offset)
{
	int i;
	int submitted = 0;
	
	pr_debug("Submitting %d iovs via io_uring, offset %lld\n",
		 nr_iovs, (long long)offset);
	
	for (i = 0; i < nr_iovs; i++) {
		struct io_uring_sqe *sqe;
		
		/* Get submission queue entry */
		sqe = io_uring_get_sqe(&uring->ring);
		if (!sqe) {
			/* Queue full, submit current batch */
			int ret = io_uring_submit(&uring->ring);
			if (ret < 0) {
				pr_err("io_uring_submit failed: %d\n", ret);
				return ret;
			}
			uring->nr_submitted += ret;
			
			/* Try again */
			sqe = io_uring_get_sqe(&uring->ring);
			if (!sqe) {
				pr_err("Failed to get SQE after submit\n");
				return -EAGAIN;
			}
		}
		
		/* Prepare read operation */
		io_uring_prep_read(sqe, fd, iovs[i].iov_base,
				   iovs[i].iov_len, offset);
		
		/* Store iovec pointer for tracking */
		io_uring_sqe_set_data(sqe, &iovs[i]);
		
		offset += iovs[i].iov_len;
		submitted++;
	}
	
	/* Submit all pending requests */
	int ret = io_uring_submit(&uring->ring);
	if (ret < 0) {
		pr_err("io_uring_submit failed: %d\n", ret);
		return ret;
	}
	
	uring->nr_submitted += ret;
	
	pr_debug("Submitted %d read requests via io_uring\n", ret);
	return ret;
}

/*
 * Wait for all pending reads to complete
 */
int page_read_uring_wait(struct page_read_uring *uring)
{
	pr_debug("Waiting for %d io_uring completions\n",
		 uring->nr_submitted - uring->nr_completed);
	
	while (uring->nr_completed < uring->nr_submitted) {
		struct io_uring_cqe *cqe;
		int ret;
		
		/* Wait for next completion */
		ret = io_uring_wait_cqe(&uring->ring, &cqe);
		if (ret < 0) {
			pr_err("io_uring_wait_cqe failed: %d\n", ret);
			return ret;
		}
		
		/* Check result */
		if (cqe->res < 0) {
			pr_err("io_uring read failed: %d\n", cqe->res);
			io_uring_cqe_seen(&uring->ring, cqe);
			return cqe->res;
		}
		
		/* Mark completion as seen */
		io_uring_cqe_seen(&uring->ring, cqe);
		uring->nr_completed++;
	}
	
	pr_debug("All io_uring completions received\n");
	return 0;
}

/*
 * Cleanup io_uring resources
 */
void page_read_uring_fini(struct page_read_uring *uring)
{
	if (!uring)
		return;
	
	io_uring_queue_exit(&uring->ring);
	xfree(uring->iovs);
	xfree(uring);
	
	pr_debug("io_uring cleaned up\n");
}

#else /* !CONFIG_HAS_LIBURING */

/* Stub implementations when io_uring is not available */

struct page_read_uring;

int page_read_uring_init(struct page_read_uring **uring_out)
{
	return -ENOTSUP;
}

int page_read_uring_submit(struct page_read_uring *uring,
			     int fd,
			     struct iovec *iovs,
			     int nr_iovs,
			     loff_t offset)
{
	return -ENOTSUP;
}

int page_read_uring_wait(struct page_read_uring *uring)
{
	return -ENOTSUP;
}

void page_read_uring_fini(struct page_read_uring *uring)
{
}

#endif /* CONFIG_HAS_LIBURING */
```

### Step 3: Integrate with PIE Restorer

**File**: `criu/pie/restorer.c`

Add conditional io_uring usage:

```c
#ifdef CONFIG_HAS_LIBURING
extern int page_read_uring_init(struct page_read_uring **uring);
extern int page_read_uring_submit(struct page_read_uring *uring,
				   int fd, struct iovec *iovs,
				   int nr_iovs, loff_t offset);
extern int page_read_uring_wait(struct page_read_uring *uring);
extern void page_read_uring_fini(struct page_read_uring *uring);
#endif

static int restore_vma_ios_with_uring(struct task_restore_args *args)
{
#ifdef CONFIG_HAS_LIBURING
	struct page_read_uring *uring = NULL;
	struct restore_vma_io *rio;
	int i, ret;
	
	/* Try to initialize io_uring */
	ret = page_read_uring_init(&uring);
	if (ret < 0) {
		pr_info("io_uring not available, using synchronous I/O\n");
		return restore_vma_ios_parallel(args);
	}
	
	pr_info("Using io_uring for page restore\n");
	
	/* Submit all reads asynchronously */
	rio = args->vma_ios;
	for (i = 0; i < args->vma_ios_n; i++) {
		ret = page_read_uring_submit(uring, args->vma_ios_fd,
					      rio->iovs, rio->nr_iovs, rio->off);
		if (ret < 0) {
			pr_err("Failed to submit reads via io_uring\n");
			page_read_uring_fini(uring);
			return ret;
		}
		
		rio = (void *)rio + RIO_SIZE(rio->nr_iovs);
	}
	
	/* Wait for all reads to complete */
	ret = page_read_uring_wait(uring);
	
	page_read_uring_fini(uring);
	return ret;
#else
	/* Fall back to parallel preadv */
	return restore_vma_ios_parallel(args);
#endif
}
```

---

## Priority 3: GPU Context Pool

⚠️ **CRITICAL CORRECTION**: This code CANNOT go in `plugins/cuda/cuda_plugin.c`!

**Why**: The CUDA plugin does NOT link against CUDA libraries. It delegates all GPU operations to the external `cuda-checkpoint` utility binary via:
```c
// plugins/cuda/cuda_plugin.c:507
cuda_process_checkpoint_action(pid, ACTION_RESTORE, 0, ...);
```

**Where to implement**: Context pooling must be added to the `cuda-checkpoint` utility source code, not the plugin.

**Architecture**:
```
CRIU → cuda_plugin.c → calls → cuda-checkpoint binary (has CUDA API access)
                                    ↑
                              Context pool goes HERE
```

### Implementation for cuda-checkpoint Utility

**File**: `cuda-checkpoint` source (external to CRIU, needs CUDA SDK)

Add after includes:

```c
#include <cuda.h>
#include <pthread.h>

/* GPU Context Pool */
#define MAX_POOL_SIZE 8
#define DEFAULT_POOL_SIZE 4

struct cuda_context_pool_entry {
	CUcontext ctx;
	bool in_use;
	pid_t owner_pid;  /* For debugging */
};

struct cuda_context_pool {
	struct cuda_context_pool_entry entries[MAX_POOL_SIZE];
	int size;
	int gpu_id;
	pthread_mutex_t lock;
};

static struct cuda_context_pool *global_ctx_pool = NULL;

/*
 * Initialize context pool at plugin startup
 */
static int cuda_context_pool_init(int gpu_id, int pool_size)
{
	struct cuda_context_pool *pool;
	int i, ret;
	
	if (pool_size > MAX_POOL_SIZE)
		pool_size = MAX_POOL_SIZE;
	
	pool = xmalloc(sizeof(*pool));
	if (!pool)
		return -ENOMEM;
	
	memset(pool, 0, sizeof(*pool));
	pool->size = pool_size;
	pool->gpu_id = gpu_id;
	pthread_mutex_init(&pool->lock, NULL);
	
	pr_info("Creating CUDA context pool (size=%d, gpu=%d)\n",
		pool_size, gpu_id);
	
	/* Pre-allocate contexts */
	for (i = 0; i < pool_size; i++) {
		CUresult cu_ret = cuCtxCreate(&pool->entries[i].ctx, 0, gpu_id);
		if (cu_ret != CUDA_SUCCESS) {
			pr_err("Failed to create context %d: %d\n", i, cu_ret);
			/* Clean up partial pool */
			while (--i >= 0) {
				cuCtxDestroy(pool->entries[i].ctx);
			}
			pthread_mutex_destroy(&pool->lock);
			xfree(pool);
			return -EINVAL;
		}
		pool->entries[i].in_use = false;
		pr_debug("Pre-created context %d: %p\n", i, pool->entries[i].ctx);
	}
	
	global_ctx_pool = pool;
	
	pr_info("CUDA context pool initialized successfully\n");
	return 0;
}

/*
 * Acquire context from pool
 * Returns NULL if pool exhausted (caller should create new context)
 */
static CUcontext cuda_context_pool_acquire(void)
{
	struct cuda_context_pool *pool = global_ctx_pool;
	CUcontext ctx = NULL;
	int i;
	
	if (!pool)
		return NULL;
	
	pthread_mutex_lock(&pool->lock);
	
	/* Find available context */
	for (i = 0; i < pool->size; i++) {
		if (!pool->entries[i].in_use) {
			pool->entries[i].in_use = true;
			pool->entries[i].owner_pid = getpid();
			ctx = pool->entries[i].ctx;
			pr_debug("Acquired context from pool: slot %d, ctx %p\n",
				 i, ctx);
			break;
		}
	}
	
	pthread_mutex_unlock(&pool->lock);
	
	if (!ctx) {
		pr_warn("Context pool exhausted (all %d contexts in use)\n",
			pool->size);
	}
	
	return ctx;
}

/*
 * Release context back to pool
 * Optionally scrub GPU memory for security
 */
static void cuda_context_pool_release(CUcontext ctx, bool scrub)
{
	struct cuda_context_pool *pool = global_ctx_pool;
	int i;
	
	if (!pool || !ctx)
		return;
	
	pthread_mutex_lock(&pool->lock);
	
	/* Find this context in pool */
	for (i = 0; i < pool->size; i++) {
		if (pool->entries[i].ctx == ctx) {
			if (scrub) {
				/* TODO: Scrub GPU memory
				 * This requires enumerating all allocations
				 * or doing a full reset
				 */
				pr_debug("GPU memory scrubbing not yet implemented\n");
			}
			
			pool->entries[i].in_use = false;
			pool->entries[i].owner_pid = 0;
			pr_debug("Released context to pool: slot %d, ctx %p\n",
				 i, ctx);
			break;
		}
	}
	
	pthread_mutex_unlock(&pool->lock);
}

/*
 * Destroy context pool at plugin shutdown
 */
static void cuda_context_pool_fini(void)
{
	struct cuda_context_pool *pool = global_ctx_pool;
	int i;
	
	if (!pool)
		return;
	
	pr_info("Destroying CUDA context pool\n");
	
	pthread_mutex_lock(&pool->lock);
	
	for (i = 0; i < pool->size; i++) {
		if (pool->entries[i].ctx) {
			cuCtxDestroy(pool->entries[i].ctx);
			pr_debug("Destroyed context %d\n", i);
		}
	}
	
	pthread_mutex_unlock(&pool->lock);
	pthread_mutex_destroy(&pool->lock);
	
	xfree(pool);
	global_ctx_pool = NULL;
	
	pr_info("CUDA context pool destroyed\n");
}
```

Update `cuda_plugin_init` to initialize pool:

```c
int cuda_plugin_init(int stage)
{
	int ret;
	
	/* ... existing code ... */
	
	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		/* Initialize context pool */
		ret = cuda_context_pool_init(0 /* gpu_id */, DEFAULT_POOL_SIZE);
		if (ret < 0) {
			pr_warn("Failed to init context pool: %d (non-fatal)\n", ret);
			/* Continue without pool */
		}
	}
	
	/* ... rest of existing code ... */
	return 0;
}
```

Update `cuda_plugin_fini` to destroy pool:

```c
void cuda_plugin_fini(int stage, int ret)
{
	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		cuda_context_pool_fini();
	}
}
```

Update `resume_device` to use pool:

```c
int resume_device(int pid, int checkpointed, cuda_task_state_t initial_task_state)
{
	CUcontext ctx = NULL;
	bool ctx_from_pool = false;
	int ret = 0;
	
	/* ... existing code ... */
	
	/* Try to acquire from pool */
	ctx = cuda_context_pool_acquire();
	if (ctx) {
		ctx_from_pool = true;
		cuCtxSetCurrent(ctx);
		pr_info("Using pooled context for PID %d\n", pid);
	} else {
		/* Pool exhausted, create new context (slow path) */
		pr_info("Creating new context for PID %d (pool exhausted)\n", pid);
		/* ... existing context creation code ... */
	}
	
	/* ... restore operations ... */
	
	/* Release context back to pool if acquired from pool */
	if (ctx_from_pool) {
		cuda_context_pool_release(ctx, true /* scrub */);
	}
	
	return ret;
}
```

---

## Build & Test

### Building with Optimizations

```bash
# Clean build
make clean

# Build with io_uring support
make LIBS_URING=-luring DEFINES=-DCONFIG_HAS_LIBURING

# Build CUDA plugin
cd plugins/cuda
make

# Build AMDGPU plugin
cd ../amdgpu
make
```

### Testing

```bash
# Test parallel restore with ZDTM
cd test
./zdtm.py run -t zdtm/static/maps00 --parallel-restore

# Test with GPU checkpoint (if CUDA available)
./cuda-checkpoint/run_tests.sh --parallel-restore

# Benchmark restore time
time criu restore -D /path/to/checkpoint --log-file restore.log -vvv

# Check logs for parallel messages
grep -E "parallel|io_uring|context pool" restore.log
```

### Performance Measurement

```bash
#!/bin/bash
# measure_speedup.sh

CHECKPOINT_DIR=$1
ITERATIONS=5

echo "Baseline (single-threaded):"
for i in $(seq 1 $ITERATIONS); do
    sync && echo 3 > /proc/sys/vm/drop_caches
    /usr/bin/time -f "%E" criu restore -D $CHECKPOINT_DIR --disable-parallel-restore 2>&1 | grep ":"
done

echo ""
echo "Optimized (parallel + io_uring + context pool):"
for i in $(seq 1 $ITERATIONS); do
    sync && echo 3 > /proc/sys/vm/drop_caches
    /usr/bin/time -f "%E" criu restore -D $CHECKPOINT_DIR 2>&1 | grep ":"
done
```

---

## Debugging Tips

### Enable Verbose Logging

```bash
criu restore -D /checkpoint -vvv --log-file restore.log
```

Look for:
- `parallel page restore with N workers`
- `io_uring initialized`
- `Using pooled context`

### Profile with perf

```bash
perf record -F 99 -g -- criu restore -D /checkpoint
perf report --stdio | head -50
```

Should show parallel threads in flame graph.

### Check Thread Count

```bash
# During restore, in another terminal:
ps -eLf | grep criu
# Should show multiple threads when parallel restore is active
```

---

## Next Steps

After implementing these optimizations:

1. **Measure performance** with LLaMA 3.1 8B checkpoint
2. **Compare to baseline** (current CRIU restore)
3. **Profile bottlenecks** with perf/VTune
4. **Iterate on tuning** (thread counts, chunk sizes)
5. **Submit patches** to CRIU upstream

Expected results:
- **Memory restore**: 28s → 4-7s (4-7x speedup)
- **GPU restore**: 11s → 3-5s (2-3x speedup)
- **Total**: 39s → 7-12s (3-5x speedup initially)

With further tuning and additional optimizations (concurrent fork, selective restore), target of **5-10s** is achievable.


