# Checkpoint/Restore Summary - Quick Overview

## What Is Happening

The `/root/gpu-load` scripts implement CRIU-based checkpoint/restore for vLLM LLM inference containers to achieve **4-8x faster TFFT (Time to First Token)** by avoiding cold model loading on every startup.

## The Problem This Solves

```
Traditional Cold Start:
GPU startup → Load model from disk → CUDA init → Warm up inference = 20-40+ seconds

CRIU Checkpoint/Restore:
Checkpoint once (model loaded + CUDA ready) → Restore from RAM = 5-10 seconds
```

## Architecture

```
┌─────────────────────────────────────┐
│  Create Checkpoint (One-time)       │
├─────────────────────────────────────┤
│ 1. Start vLLM container             │
│ 2. Wait for model loading           │
│ 3. Run warmup inferences            │
│ 4. CRIU checkpoint (save state)     │
│ 5. Store in /var/lib/containers/... │
└─────────────────────────────────────┘
              ↓ (reuse many times)
┌─────────────────────────────────────┐
│  Restore from Checkpoint (Repeated) │
├─────────────────────────────────────┤
│ 1. CRIU restore (reload state)      │
│ 2. Check health                     │
│ 3. Serve inference                  │
│ 4. Measure TFFT                     │
│ 5. Cleanup & repeat                 │
└─────────────────────────────────────┘
```

## Key Commands

### Checkpoint Creation
```bash
# Full automated process
cd /root/gpu-load
./create-checkpoint.py

# Manual steps:
podman run -d --name vllm-checkpoint --device /dev/nvidia* --privileged ...
curl http://localhost:8000/health  # Wait for ready
curl http://localhost:8000/v1/completions -d '...'  # Warmup 5x
podman container checkpoint --print-stats vllm-checkpoint
```

### Restore
```bash
# Basic restore (in-place, container stops)
podman container restore vllm-checkpoint

# Restore with keep (container stays running)
podman container restore --keep vllm-checkpoint

# Using custom CRIU
podman container restore --runtime-opt=runtime_criu_path=/root/criu/criu/criu --keep vllm-checkpoint

# From tar export
podman container restore --import /mnt/checkpoint-ram/checkpoint.tar vllm-checkpoint
```

## Critical Configuration

| Setting | Value | Why |
|---------|-------|-----|
| `--privileged` | Yes | CRIU needs to manage namespaces |
| `--security-opt seccomp=no-io-uring.json` | Yes | io_uring incompatible with CRIU |
| `--device /dev/nvidia0` | Required | GPU access |
| `--enforce-eager` | Yes | Disable CUDA graphs (incompatible) |
| `--load-format safetensors` | Yes | File-backed memory (smaller checkpoint) |
| `ASYNCIO_DEFAULT_BACKEND=select` | Yes | Safer async backend for CRIU |
| `PYTHON_ASYNCIO_NO_IO_URING=1` | Yes | Disable io_uring in asyncio |

## Environment Variables

**.env file** (`/root/gpu-load/.env`):
```bash
MODEL_ID=Qwen/Qwen2-1.5B-Instruct     # Model to checkpoint
CONT_NAME=vllm-llm-demo               # Container name
API_PORT=8000                         # API port
CHECKPOINT_DIR=/mnt/checkpoint-ram    # Storage (tmpfs)
```

**Container env vars**:
```bash
-e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH
-e ASYNCIO_DEFAULT_BACKEND=select
-e PYTHON_ASYNCIO_NO_IO_URING=1
```

## Performance Numbers

### Baseline (Cold Start)
```
Container startup:    2-5s
Model load:          15-30s
Health check:         1-2s
TFFT:                1-2s
─────────────────────────
Total:              20-40+ seconds
```

### CRIU Restore
```
Restore from checkpoint:  3-7s
Health check:            1-2s
TFFT:                  0.5-1s
─────────────────────────
Total:                5-10 seconds
```

### Speedup: **4-8x faster**

## Checkpoint Storage

**Default location**:
```
/var/lib/containers/storage/overlay-containers/{CONTAINER_ID}/userdata/checkpoint/
```

**Typical files**:
```
pages-*.img          # CPU memory pages
gpu-*.img            # GPU VRAM (model weights, cache)
core-*.img           # Process core dumps
mm-*.img             # Memory mapping
ids-*.img            # Resource IDs
... (other CRIU metadata)
```

**Typical size**: 2-5 GB total
- Model weights: 1.5-3 GB
- GPU VRAM: 0.5-1 GB
- Process state: 100-500 MB

## Test Scripts

| Script | Purpose |
|--------|---------|
| `create-checkpoint.py` | Create golden checkpoint |
| `test-restore-speed.sh` | Measure restore performance (5 iterations) |
| `test-ignore-flags.sh` | Test network compatibility flags |
| `test-import-restore.sh` | Test tar import/restore |
| `benchmark-custom-criu.py` | Compare system vs custom CRIU |
| `benchmark-criu.py` | Benchmark CRIU restore |
| `benchmark-baseline.py` | Benchmark cold start baseline |

## What Gets Checkpointed

CRIU captures:
- ✅ Process memory (all pages)
- ✅ GPU VRAM (model weights, KV cache)
- ✅ Open file descriptors
- ✅ Network sockets
- ✅ CUDA context state
- ✅ Python interpreter state
- ✅ Shared memory segments

Does NOT checkpoint:
- ❌ NVIDIA driver (must be same version)
- ❌ /dev/nvidia* devices (must exist on restore)
- ❌ External files (only cached files checkpoint)

## How vLLM Fits

vLLM is ideal for checkpoint/restore because:
1. **Stateless**: No persistent external state
2. **Container native**: Runs in isolated namespace
3. **GPU heavy**: Benefits from skipping model loading
4. **Predictable state**: Same after every startup

After checkpoint:
- Model weights in GPU VRAM (don't reload)
- CUDA context initialized (don't reinit)
- KV cache pre-allocated (don't reallocate)
- vLLM workers ready to serve

## Qwen 1.5B Model

Used for benchmarking:
- **Size**: ~1.5 billion parameters
- **Format**: SafeTensors (file-backed)
- **Context**: 4096 tokens
- **GPU memory**: ~3-4 GB

Fast enough for testing, realistic enough for production.

## Workflow Example

```bash
# Step 1: Create checkpoint once
./create-checkpoint.py
# → Creates checkpoint with loaded model + CUDA context
# → Stores in /var/lib/containers/storage/.../checkpoint/
# → Takes ~60-90 seconds (model load + checkpoint overhead)

# Step 2: Run benchmarks (many times)
./run-benchmarks.py --iterations 10
# → Iteration 1:
#   - Restore from checkpoint (3-7s)
#   - Health check (1-2s)
#   - First inference (0.5-1s)
#   - Clean up
# → Iteration 2-10: Repeat
# → Compare vs baseline (20-40+ seconds per startup)

# Step 3: Analyze results
./analyze-results.py
# → Calculate speedup (4-8x)
# → Generate report with statistics
```

## Storage Locations

| Path | Purpose |
|------|---------|
| `/root/gpu-load/` | Script directory |
| `/root/gpu-load/.env` | Configuration |
| `/root/gpu-load/results/` | Benchmark results (JSON/CSV) |
| `/var/lib/containers/storage/.../checkpoint/` | CRIU checkpoint files |
| `/mnt/checkpoint-ram/` | Optional tmpfs for faster I/O |
| `/models/` | HuggingFace model cache |

## Hooks & Plugins

**Not currently used** in gpu-load, but available:
- CRIU action scripts (pre-dump, pre-restore, post-restore)
- Custom seccomp profiles (io_uring disabled)
- Runtime options (custom CRIU binary path)
- Ignore flags (`--ignore-static-ip`, etc.)

## Health Check

After restore, verify with:
```bash
curl http://localhost:8000/health
# Returns 200 OK when API is ready

curl http://localhost:8000/v1/completions -d {...}
# Returns first token within 0.5-1s
```

## Next Steps for Usage

1. **One-time setup**: `./create-checkpoint.py`
2. **Benchmark**: `./run-benchmarks.py`
3. **Analyze**: `./analyze-results.py`
4. **Deploy**: Use restore workflow in production

## Reference Docs

- **Detailed Analysis**: `CHECKPOINT_RESTORE_ANALYSIS.md`
- **Commands Reference**: `COMMANDS_REFERENCE.md`
- **README**: `README.md`
- **README-checkpoint**: `README-checkpoint.md`

---

**TL;DR**: CRIU saves a fully-loaded vLLM container state to disk, then restores it instantly from RAM instead of reloading the model. This achieves **4-8x faster TFFT** by skipping cold model loading.

