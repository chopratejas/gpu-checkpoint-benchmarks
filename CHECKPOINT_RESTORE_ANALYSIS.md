# GPU-Load Checkpoint/Restore Workflow Analysis

## Executive Summary

The `/root/gpu-load` scripts implement a comprehensive CRIU-based checkpoint/restore system for vLLM LLM inference containers. The workflow captures fully-initialized Qwen LLM models with CUDA contexts in checkpoints stored in DRAM, enabling sub-second restore times for low-latency inference serving.

---

## 1. CHECKPOINT CREATION WORKFLOW

### 1.1 Entry Point: `create-checkpoint.py`

**Purpose**: Create a golden checkpoint of vLLM container after warmup

**Execution Flow**:
```
1. Load environment config (.env)
2. Remove existing containers
3. Start fresh vLLM container with GPU access
4. Wait for health check (model fully loaded)
5. Run 5 warmup inference requests (CUDA optimization)
6. Create checkpoint via CRIU
7. Verify checkpoint exists and measure size
```

### 1.2 Container Startup Command

**Key podman run flags for GPU checkpoint compatibility**:

```bash
podman run -d \
  --name vllm-checkpoint \
  
  # GPU Device Access
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  
  # Privileged Mode (required for CRIU GPU support)
  --privileged \
  
  # Security/Compatibility
  --security-opt=label=disable \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  
  # Memory Configuration
  --shm-size 8g \
  
  # Environment for CUDA/asyncio compatibility
  -e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  
  # Volumes
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  
  # Container Image
  docker.io/vllm/vllm-openai:latest \
  
  # vLLM Server Arguments
  --model Qwen/Qwen2-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 1024 \
  --enforce-eager \
  --load-format safetensors \
  --trust-remote-code
```

**Critical Flags for CRIU GPU Checkpoint**:
- `--privileged`: Allows CRIU to inject parasite code and manage namespaces
- `--security-opt seccomp=...`: Disables io_uring (incompatible with CRIU)
- `--device /dev/nvidia*`: GPU device mapping
- `--enforce-eager`: Disables CUDA graphs (CRIU incompatible)
- `--load-format safetensors`: Uses file-backed memory (optimizes checkpoint size)
- `ASYNCIO_DEFAULT_BACKEND=select`: Select backend for asyncio (safer for CRIU)

### 1.3 Health Check Phase

**Function**: `utils.wait_for_health(port=8000, timeout=300)`

**Command**:
```bash
curl -s http://localhost:8000/health
```

**Polling**: Every 2 seconds, up to 5 minutes timeout
**Return**: Time in seconds until health check passes

**Why needed**: Ensures model is fully loaded into VRAM before checkpoint

### 1.4 Warmup Inference Requests

**Function**: `run_warmup_requests(num_requests=5, port=8000)`

**Example Command** (via HTTP API):
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "stream": true,
    "temperature": 0.7
  }'
```

**Purpose**: 
- Warm up CUDA kernels
- Initialize KV cache allocations
- Ensure all GPU optimizations are in place

**Metrics Captured**:
- TFFT (Time to First Token)
- Total inference time
- Tokens generated

### 1.5 Checkpoint Creation Command

**Podman Checkpoint**:
```bash
podman container checkpoint \
  --print-stats \
  vllm-checkpoint
```

**NOT Using**:
- `--export /path/to/tar`: Does NOT export to tar (uses local storage)
- `--keep`: Does NOT keep container running

**Storage Location**:
```
/var/lib/containers/storage/overlay-containers/{container_id}/userdata/checkpoint/
```

**What is Checkpointed**:
- Process memory pages
- GPU VRAM contents (model weights, KV cache, etc.)
- Open file descriptors
- Network sockets
- CUDA context state
- Python interpreter state

**Checkpoint Contents** (CRIU image files):
```
checkpoint/
├── pages-*.img        # Memory pages (CPU RAM)
├── gpu-*.img          # GPU VRAM pages
├── core-*.img         # Process core dumps
├── mm-*.img           # Memory mapping info
├── ids-*.img          # Resource identifiers
├── netdev-*.img       # Network devices
├── tty-*.img          # Terminal state
└── ... (other CRIU metadata)
```

### 1.6 Checkpoint Verification

**Location Check**:
```bash
CONTAINER_ID=$(podman inspect --format '{{.Id}}' vllm-checkpoint)
CKPT_DIR="/var/lib/containers/storage/overlay-containers/${CONTAINER_ID}/userdata/checkpoint"
du -sh $CKPT_DIR  # Check size
```

**Typical Checkpoint Sizes**:
- Model weights (file-backed): ~1.5-3 GB
- GPU VRAM (KV cache, runtime): ~0.5-1 GB
- Process state: ~100-500 MB
- **Total**: 2-5 GB

---

## 2. RESTORE WORKFLOW

### 2.1 Basic Restore Command

**Standard Restore** (in-place, don't keep checkpoint):
```bash
podman container restore vllm-checkpoint
```

**Restore with Keep** (keep container running):
```bash
podman container restore --keep vllm-checkpoint
```

**Restore with Ignore Flags** (for network compatibility):
```bash
podman container restore --keep \
  --ignore-static-ip \
  --ignore-static-mac \
  vllm-checkpoint
```

### 2.2 Restore with Custom CRIU Binary

**Using System CRIU**:
```bash
podman container restore --keep vllm-checkpoint
```

**Using Custom CRIU Build** (from /root/criu):
```bash
podman container restore \
  --runtime-opt=runtime_criu_path=/root/criu/criu/criu \
  --keep \
  vllm-checkpoint
```

### 2.3 Restore Import from Tar Export

**Export Checkpoint to Tar**:
```bash
podman container checkpoint --export /mnt/checkpoint-ram/checkpoint.tar vllm-checkpoint
```

**Import and Restore from Tar**:
```bash
podman container restore \
  --import /mnt/checkpoint-ram/checkpoint.tar \
  vllm-checkpoint
```

**Use Cases**:
- Moving checkpoints to faster storage (ramdisk)
- Distributing checkpoints across systems
- Compressing with zstd/xz

### 2.4 Restore Performance Timing

**From benchmark-custom-criu.py**:

```python
# Phases measured:
1. restore_time     # CRIU restore duration (actual restore)
2. health_time      # Time until API health check passes
3. inference_time   # Time to run first inference
4. total_time       # restore_time + health_time

# Typical values (Qwen 1.5B on NVIDIA GPU):
restore_time:      3-7 seconds
health_time:       1-3 seconds  
total_time:        4-10 seconds
```

---

## 3. CONFIGURATION & ENVIRONMENT VARIABLES

### 3.1 .env Configuration File

**Location**: `/root/gpu-load/.env`

**Key Variables**:

```bash
# Model Configuration
MODEL_ID=Qwen/Qwen2-1.5B-Instruct         # HuggingFace model ID
MAX_MODEL_LEN=4096                        # Context length
GPU_MEMORY_UTIL=0.90                      # GPU memory utilization %

# Container Configuration  
CONT_NAME=vllm-llm-demo                   # Container name
VLLM_IMAGE=docker.io/vllm/vllm-openai:latest  # Image
API_PORT=8000                             # API port

# Checkpoint Configuration
CHECKPOINT_DIR=/mnt/checkpoint-ram        # Checkpoint storage (tmpfs/ramdisk)
CKPT_PATH=/mnt/checkpoint-ram/checkpoint.tar  # Tar export path

# Directory Paths
RESULTS_DIR=./results/20251028_224932     # Benchmark results
NVIDIA_LIBS_PATH=/opt/nvidia-libs        # NVIDIA libs mount
MODELS_CACHE=/models                      # Model cache directory

# Performance Configuration
WARMUP_REQUESTS=5                         # Warmup iterations
HEALTH_CHECK_TIMEOUT=300                  # Health check timeout (seconds)
```

### 3.2 Container Environment Variables

**Set in create-checkpoint.py**:

```bash
LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH
  # CUDA runtime libraries

ASYNCIO_DEFAULT_BACKEND=select
  # Select async backend (safer than io_uring for CRIU)

PYTHON_ASYNCIO_NO_IO_URING=1
  # Explicitly disable io_uring for asyncio
```

---

## 4. ACTUAL COMMANDS EXECUTED

### 4.1 Checkpoint Creation Commands

**Full sequence**:

```bash
# 1. Remove existing container
podman rm -f vllm-checkpoint

# 2. Start container
podman run -d --name vllm-checkpoint \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --privileged --security-opt=label=disable \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --shm-size 8g \
  -e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 1024 \
  --enforce-eager \
  --load-format safetensors \
  --trust-remote-code

# 3. Health check (loop until 200 OK)
curl -s http://localhost:8000/health

# 4. Warmup inference (5x)
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2-1.5B-Instruct", "prompt": "...", "max_tokens": 50, ...}'

# 5. Create checkpoint
podman container checkpoint --print-stats vllm-checkpoint

# 6. Verify checkpoint
CONTAINER_ID=$(podman inspect --format '{{.Id}}' vllm-checkpoint)
ls -lh /var/lib/containers/storage/overlay-containers/$CONTAINER_ID/userdata/checkpoint/
```

### 4.2 Restore Commands

**Test-Restore-Speed Script** (`test-restore-speed.sh`):

```bash
#!/bin/bash
# Loop 5 times:
for i in 1 2 3 4 5; do
  START=$(date +%s%N)
  
  # Core restore operation
  podman container restore --keep vllm-llm-demo
  
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "Restore time: ${ELAPSED}s"
  
  # Wait for health
  sleep 2
  timeout 5 curl -s http://localhost:8000/health > /dev/null
  
  # Stop for next iteration
  podman stop vllm-llm-demo 2>/dev/null
  sleep 2
done
```

**Ignore-Flags Test** (`test-ignore-flags.sh`):

```bash
# With compatibility flags
podman container restore --keep \
  --ignore-static-ip \
  --ignore-static-mac \
  vllm-llm-demo
```

**Import Test** (`test-import-restore.sh`):

```bash
# Assume checkpoint exported to tar
podman container restore \
  --import /mnt/checkpoint-ram/checkpoint-zstd.tar \
  vllm-llm-demo
```

---

## 5. CHECKPOINT STORAGE

### 5.1 Checkpoint Directories

**Podman Internal Storage** (default):
```
/var/lib/containers/storage/overlay-containers/{CONTAINER_ID}/userdata/checkpoint/
```

**DRAM-Based (tmpfs/ramdisk)**:
```
/mnt/checkpoint-ram/
```
- Size: 16GB (configured in setup.sh)
- Mount type: tmpfs
- Purpose: Fast checkpoint storage and export

### 5.2 Checkpoint File Export

**Export with tar**:
```bash
podman container checkpoint --export /mnt/checkpoint-ram/checkpoint.tar vllm-checkpoint
```

**Export with compression**:
```bash
# Manual compression of tar
tar czf /mnt/checkpoint-ram/checkpoint-gzip.tar.gz checkpoint/
tar -I zstd -cf /mnt/checkpoint-ram/checkpoint-zstd.tar checkpoint/
```

**Typical Checkpoint Artifacts**:
- `checkpoint.tar`: Uncompressed (~2-5 GB)
- `checkpoint.tar.gz`: gzip (~1-2 GB)
- `checkpoint-zstd.tar`: zstd (~800MB-1.5GB)

---

## 6. HOOKS & CUSTOM PLUGINS

### 6.1 CRIU Hooks

**Not explicitly configured in gpu-load scripts**, but CRIU supports:

- Pre-dump hooks (before checkpoint)
- Pre-restore hooks (before restore)  
- Post-restore hooks (after restore)

These would be passed via CRIU command line options, e.g.:
```bash
--action-script=/path/to/hook.sh
```

### 6.2 Seccomp Configuration

**Custom Seccomp Profile** (`/etc/containers/seccomp.d/no-io-uring.json`):

Purpose: Disable io_uring (incompatible with CRIU checkpoint/restore)

Key restriction:
```json
{
  "syscalls": [
    {
      "names": ["io_uring_setup", "io_uring_enter", "io_uring_register"],
      "action": "SCMP_ACT_ERRNO"
    }
  ]
}
```

### 6.3 GPU-Specific Considerations

**CUDA Context Checkpoint Requirements**:

1. **No CUDA Graphs**: `--enforce-eager` flag disables CUDA graphs (incompatible with checkpoint)

2. **GPU Memory State**: Checkpointed as part of process memory pages

3. **NVIDIA Driver State**: Not directly checkpointed; requires compatible driver on restore

4. **Device Access**: Relies on `/dev/nvidia*` devices remaining available

---

## 7. BENCHMARK WORKFLOW

### 7.1 Baseline Benchmark (`benchmark-baseline.py`)

**Measures cold-start time**:

```
T0: Start timer
  ↓
T0-T1: podman run (container startup)
  ↓
T1-T2: vLLM model loading (from disk to GPU)
  ↓
T2-T3: Health check passes (API ready)
  ↓
T3-T4: First inference token received (TFFT)
  ↓
T4: Full response complete
```

**Typical timings**:
- Container startup: 2-5s
- Model load to GPU: 15-30s
- Health check: 1-2s
- TFFT: 1-2s
- **Total**: 20-40+ seconds

### 7.2 CRIU Benchmark (`benchmark-criu.py`)

**Measures checkpoint-restore time**:

```
T0: Start timer
  ↓
T0-T1: podman restore from checkpoint
  ↓
T1-T2: CRIU restores GPU VRAM
  ↓
T2-T3: Health check passes (API ready)
  ↓
T3-T4: First inference token received (TFFT)
  ↓
T4: Full response complete
```

**Typical timings**:
- CRIU restore: 3-7s
- Health check: 1-2s
- TFFT: 0.5-1s
- **Total**: 5-10 seconds

**Speedup**: ~4-8x faster than cold start

---

## 8. TEST SCRIPTS

### 8.1 `test-restore-speed.sh`

**Purpose**: Measure restore performance across multiple iterations

**Metrics**: 
- Individual restore times
- Health check status
- Raw execution time using nanoseconds

### 8.2 `test-ignore-flags.sh`

**Purpose**: Test CRIU compatibility flags for network configs

**Flags tested**:
- `--ignore-static-ip`: Don't restore static IP
- `--ignore-static-mac`: Don't restore MAC address

**Comparison**: Baseline vs. with flags

### 8.3 `test-import-restore.sh`

**Purpose**: Test checkpoint import from tar archive

**Workflow**:
1. Remove container
2. Restore from tar import
3. Check health
4. Measure time
5. Clean up

**Use Case**: Simulates distributed checkpoint scenarios

---

## 9. KEY INSIGHTS

### 9.1 Why This Works for vLLM

1. **State Isolation**: vLLM runs in containers (isolated processes/namespaces)
2. **GPU Memory Capture**: CRIU can checkpoint GPU VRAM (NVIDIA driver support)
3. **Stateless by Design**: vLLM doesn't maintain external state after startup
4. **Warm Cache**: Checkpoint captures pre-initialized model and KV cache

### 9.2 Performance Bottlenecks

1. **GPU VRAM Copy**: Checkpoint restore must copy GPU VRAM back to device (~100-500MB/s)
2. **CUDA Context Restoration**: CRIU must reinitialize CUDA context (~0.5-1s)
3. **File-backed Memory**: Model weights on tmpfs (faster than NVMe)
4. **Process Tree Complexity**: vLLM worker processes add overhead

### 9.3 Optimization Opportunities

From `checkpoint-analysis.py`:

```python
# Checkpoint size breakdown:
- Memory pages: 30-50% (optimize with load-format)
- GPU data: 20-40% (GPU VRAM, KV cache)
- Process state: 10-20% (core dumps, metadata)
- Filesystem: 5-10% (reduce with --ignore-rootfs)
```

---

## 10. COMPLETE WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU-Load Checkpoint/Restore                      │
└─────────────────────────────────────────────────────────────────────┘

CHECKPOINT PHASE
═══════════════════════════════════════════════════════════════════════

1. Remove existing container
   └─ podman rm -f vllm-checkpoint

2. Start fresh vLLM container
   └─ podman run -d --device /dev/nvidia* --privileged ...
      └─ Container starts with fresh GPU memory
      └─ Model weights loaded from /models cache
      └─ vLLM server listening on port 8000

3. Wait for health check  
   └─ GET http://localhost:8000/health (poll every 2s, timeout 5m)
      └─ Model fully loaded into GPU VRAM
      └─ CUDA context initialized
      └─ vLLM server ready

4. Run warmup inferences (5 requests)
   └─ POST http://localhost:8000/v1/completions
      └─ Warm up CUDA kernels
      └─ Initialize KV cache
      └─ Measure TFFT, tokens/sec

5. Create CRIU checkpoint
   └─ podman container checkpoint --print-stats vllm-checkpoint
      └─ CRIU injects parasite code
      └─ Captures all process memory, GPU VRAM, file descriptors
      └─ Stores at /var/lib/containers/storage/overlay-containers/{ID}/userdata/checkpoint/

6. Verify checkpoint
   └─ Check checkpoint directory exists
   └─ Measure size (typically 2-5 GB)
   └─ Log creation time and metrics


RESTORE PHASE (Multiple iterations)
═══════════════════════════════════════════════════════════════════════

1. Clean up any prior container
   └─ podman rm -f vllm-checkpoint

2. Restore from checkpoint
   └─ podman container restore --keep vllm-checkpoint
      └─ CRIU restores all processes
      └─ Restores GPU VRAM to device
      └─ Reinitializes CUDA context
      └─ Container comes up with same state as checkpoint

3. Wait for health check
   └─ GET http://localhost:8000/health
      └─ Verify API is responding (usually immediate)

4. Run inference and measure TFFT
   └─ POST http://localhost:8000/v1/completions
      └─ Measure time to first token
      └─ Record latency metrics

5. Stop container
   └─ podman stop vllm-checkpoint

6. Repeat (or clean up)
   └─ podman rm -f vllm-checkpoint


PERFORMANCE COMPARISON
═══════════════════════════════════════════════════════════════════════

Baseline (Cold Start):        20-40+ seconds
  - Container startup:         2-5s
  - Model load:               15-30s
  - Health check:              1-2s
  - TFFT:                      1-2s

CRIU (Warm Start):            5-10 seconds
  - CRIU restore:              3-7s
  - Health check:              1-2s
  - TFFT:                    0.5-1s

Speedup: 4-8x faster
```

---

## Summary Table

| Aspect | Details |
|--------|---------|
| **Container** | vLLM with Qwen 1.5B-Instruct |
| **Checkpoint Command** | `podman container checkpoint [--print-stats] {container}` |
| **Restore Command** | `podman container restore [--keep] [--ignore-*] {container}` |
| **Storage Location** | `/var/lib/containers/storage/overlay-containers/{ID}/userdata/checkpoint/` |
| **Checkpoint Size** | 2-5 GB (model + GPU VRAM + process state) |
| **Health Check** | `GET http://localhost:8000/health` (poll every 2s) |
| **Warmup Requests** | 5 inference requests pre-checkpoint |
| **Key Env Vars** | `ASYNCIO_DEFAULT_BACKEND=select`, `PYTHON_ASYNCIO_NO_IO_URING=1` |
| **CRIU Flags** | `--privileged`, custom seccomp (no io_uring) |
| **vLLM Flags** | `--enforce-eager`, `--load-format safetensors` |
| **Typical Restore Time** | 5-10 seconds (vs 20-40+ seconds cold start) |
| **Speedup** | 4-8x faster TFFT |

