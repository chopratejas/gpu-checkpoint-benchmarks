# CRIU GPU Benchmark Suite - Current State

**Date**: October 28, 2025
**System**: Lambda Cloud GPU Instance (NVIDIA A10, 23GB VRAM)
**Location**: `/root/gpu-load/`

---

## 🎯 Project Goal

Benchmark Cold Start TFFT (Time to First Token) comparing:
- **Baseline**: Traditional cold start (model load from NVMe → GPU VRAM)
- **CRIU**: Checkpoint/restore from DRAM (tmpfs)

Target: Achieve **InferX/Modal-level performance** with sub-3s TFFT for small models.

---

## ✅ What's Working

### 1. Environment Setup ✓
```bash
# Completed successfully
sudo bash setup.sh
```

**Status**: COMPLETE
- uv package manager: v0.9.5 installed
- tmpfs ramdisk: 16GB mounted at `/mnt/checkpoint-ram`
- CRIU: v4.1.1 verified with GPU support
- Podman: v4.9.3 verified
- GPU: NVIDIA A10 (23GB VRAM, Driver 570.158.01)
- Seccomp profile: Created at `/etc/containers/seccomp.d/no-io-uring.json`
- Environment: `.env` file generated

### 2. Utilities Module ✓
```bash
export PATH="$HOME/.local/bin:$PATH"
./test_utils.py
```

**Status**: ALL TESTS PASSED
- `get_timestamp_ns()` - Nanosecond precision timing ✓
- `log_metric()` - Structured logging ✓
- `load_env()` - Environment loading ✓
- `run_command()` - Shell execution with timing ✓
- `cleanup_containers()` - Container cleanup ✓
- `wait_for_health()` - Health check polling (requires vLLM)
- `send_inference()` - TFFT measurement (requires vLLM)

### 3. Checkpoint Creation ✓
```bash
./create-checkpoint.py
```

**Status**: COMPLETE - Checkpoint created successfully

**Results**:
- Container started: vllm-llm-demo
- Health check passed: 58.20s
- Warmup inferences: 5 requests completed
  - TFFT range: 0.040-0.081s
  - Tokens/second: 81-87 tok/s
- **Checkpoint created**: `/mnt/checkpoint-ram/checkpoint.tar`
- **Checkpoint size**: 3.6 GB (in tmpfs RAM)
- **Total time**: 125.70s

**Location**: `/mnt/checkpoint-ram/checkpoint.tar` (3.6GB in RAM)

### 4. Baseline Benchmark ✓
```bash
./benchmark-baseline.py
```

**Status**: COMPLETE - Baseline measured successfully

**Results**: `results/20251028_224932/baseline_20251028_225425.json`
```json
{
    "timestamp": "2025-10-28T22:54:25.821129",
    "type": "baseline",
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "container_start_time": 0.24152,
    "model_load_time": 58.910359,
    "inference_tfft": 0.076246,
    "total_cold_start_tfft": 59.228125,
    "full_response_time": 59.760973
}
```

**Key Metrics**:
- Container start: **0.24s**
- Model load (to health): **58.91s**
- Inference TFFT: **0.08s**
- **Total Cold Start TFFT: 59.23 seconds** 🎯

**Inference Response**: Successfully generated haiku about GPUs

---

## ❌ Critical Problem: CRIU Restore Performance

### Issue: CRIU Restore is SLOWER than Baseline

**Manual Test Results**:
```bash
time podman container restore --import=/mnt/checkpoint-ram/checkpoint.tar --ignore-volumes
# Result: 2m4.479s (124.5 seconds)
```

**Performance Comparison**:
| Method | Time | Result |
|--------|------|--------|
| Baseline Cold Start | 59.2s | ✅ Works |
| CRIU Restore | 124.5s | ❌ **2.1x SLOWER!** |

### What We Verified

1. ✅ Checkpoint is in tmpfs (RAM) - not disk I/O issue
2. ✅ Checkpoint file exists and is correct size (3.6GB)
3. ✅ Container DOES work after restore
4. ✅ API responds and serves inference successfully
5. ✅ Commands match user's working setup exactly:
   ```bash
   # Checkpoint
   podman container checkpoint --export=/mnt/checkpoint-ram/checkpoint.tar --ignore-volumes vllm-llm-demo

   # Restore
   podman container restore --import=/mnt/checkpoint-ram/checkpoint.tar --ignore-volumes
   ```

### Why Is Restore So Slow?

**Hypothesis**: GPU memory restoration bottleneck
- **11GB of GPU VRAM** needs to be restored
- CRIU v4.1.1 may have slow/immature GPU restore implementation
- Copy path: Checkpoint (RAM) → CPU RAM → GPU VRAM (inefficient)
- **GPU memory restoration is taking longer than loading model from scratch**

### System Configuration

**CRIU**:
```bash
criu --version
# Version: 4.1.1

criu check --feature mem_dirty_track
# mem_dirty_track is supported
```

**Checkpoint Storage**:
```bash
df -h /mnt/checkpoint-ram
# Filesystem      Size  Used Avail Use% Mounted on
# tmpfs            16G  3.6G   13G  23% /mnt/checkpoint-ram

mount | grep checkpoint-ram
# tmpfs on /mnt/checkpoint-ram type tmpfs (rw,relatime,size=16777216k,inode64)
```

---

## 📊 Current Test Results

### Checkpoint Creation Timeline
1. Container start: ~2s
2. Model loading: ~58s
3. Health check ready: 58.20s total
4. Warmup inferences (5x): ~3s
5. Checkpoint creation: 61.95s
6. **Total**: 125.70s

### Baseline Cold Start Timeline
1. Container start: 0.24s
2. Model load (NVMe → CPU → GPU): 58.91s
3. API ready (health check passes)
4. First inference TFFT: 0.08s
5. **Total TFFT**: 59.23s

### CRIU Restore Timeline (PROBLEMATIC)
1. Restore from tmpfs: **124.5s** ⚠️
2. Container restored successfully
3. API responds (tested with curl)
4. Inference works correctly
5. **Total**: ~125s (no speedup!)

---

## 📁 Files Created

### Core Scripts
- `setup.sh` (8.7KB) - Environment initialization ✓
- `utils.py` (12KB) - Shared utilities with PEP 723 ✓
- `create-checkpoint.py` (11KB) - Checkpoint creation ✓
- `benchmark-baseline.py` (16KB) - Baseline benchmark ✓
- `benchmark-criu.py` (9.2KB) - CRIU benchmark (needs investigation)
- `run-benchmarks.py` (11KB) - Orchestration script (not tested yet)
- `analyze-results.py` (17KB) - Analysis script (not tested yet)

### Documentation
- `README.md` (17KB) - Complete documentation
- `QUICKSTART.md` (3.3KB) - Quick start guide
- `CURRENT_STATE.md` (this file) - Current state capture

### Test & Support Files
- `test_utils.py` (2.6KB) - Utility tests ✓
- `.env` - Environment configuration ✓

### Results
- `results/20251028_224932/baseline_20251028_225425.json` - Baseline results ✓
- Checkpoint: `/mnt/checkpoint-ram/checkpoint.tar` (3.6GB in RAM) ✓

---

## 🔍 Investigation Needed

### Critical Questions

1. **What TFFT timing did you achieve** with your working CRIU setup on this Lambda system?
   - Was it faster than baseline?
   - How much faster?

2. **Did you use any special CRIU flags** beyond `--ignore-volumes`?
   - Any optimization flags?
   - Different checkpoint options?

3. **Did you checkpoint at a different point**?
   - Before model fully loads?
   - After partial loading?
   - Different warmup strategy?

4. **CRIU version discrepancy**?
   - Current: v4.1.1
   - Is there a newer version available?
   - Are there known GPU restore performance issues in 4.1.1?

5. **Container name consistency**?
   - Should we use `--name` flag on restore?
   - Does name matching matter for GPU restore?

### Areas to Explore

1. **CRIU GPU Restore Optimization**
   ```bash
   # Check for additional CRIU options
   criu restore --help | grep -i gpu
   podman container restore --help
   ```

2. **Checkpoint at Different Stage**
   - Checkpoint before full model load?
   - Checkpoint after engine init but before full warmup?
   - Smaller GPU memory footprint = faster restore?

3. **Alternative Approaches**
   - Use CRIU directly (not through podman)?
   - Use `--leave-running` flag?
   - Use `--tcp-established` for network sockets?

4. **System-Level Optimizations**
   - NUMA node pinning for tmpfs?
   - GPU P-states or clocking?
   - PCIe settings?

---

## 🚀 How to Continue

### Resume Testing

1. **Source environment**:
   ```bash
   cd /root/gpu-load
   export PATH="$HOME/.local/bin:$PATH"
   source .env
   ```

2. **Check checkpoint exists**:
   ```bash
   ls -lh /mnt/checkpoint-ram/checkpoint.tar
   # Should show 3.6G
   ```

3. **Test restore manually**:
   ```bash
   podman rm -f vllm-llm-demo
   time podman container restore --import=/mnt/checkpoint-ram/checkpoint.tar --ignore-volumes
   ```

4. **Check if container works**:
   ```bash
   sleep 5
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"Qwen/Qwen2-1.5B-Instruct","prompt":"Say hello","max_tokens":10}'
   ```

5. **Run CRIU benchmark** (if restore is faster):
   ```bash
   ./benchmark-criu.py
   ```

6. **Run full benchmark suite**:
   ```bash
   ./run-benchmarks.py --iterations 3
   ```

7. **Analyze results**:
   ```bash
   ./analyze-results.py
   ```

### Cleanup Commands

```bash
# Remove all containers
podman rm -f $(podman ps -aq)

# Check tmpfs usage
df -h /mnt/checkpoint-ram

# Check for background processes
ps aux | grep -E "podman|criu|vllm"

# Kill any stuck processes
pkill -9 podman
```

---

## 🐛 Known Issues

### 1. CRIU Restore Performance (CRITICAL)
**Status**: UNRESOLVED
**Symptom**: Restore takes 124.5s vs baseline 59.2s
**Impact**: CRIU is 2.1x slower than cold start - defeats the purpose
**Next Steps**: Need to understand why GPU restore is so slow

### 2. Health Check After Restore
**Status**: WORKS but unclear timing
**Symptom**: API doesn't respond immediately after restore
**Impact**: Unclear how long reinitialization takes
**Next Steps**: Need to measure T2 (health check pass) after restore

### 3. Old Checkpoint in /dev/shm
**Status**: MINOR
**Note**: There's an old checkpoint at `/dev/shm/vllm-ckpt.tar` (3.6GB) from earlier tests
**Action**: Can be deleted if space is needed

---

## 📝 Configuration

### Current .env Settings
```bash
MODEL_ID=Qwen/Qwen2-1.5B-Instruct
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90
CONT_NAME=vllm-llm-demo
VLLM_IMAGE=docker.io/vllm/vllm-openai:latest
API_PORT=8000
CHECKPOINT_DIR=/mnt/checkpoint-ram
CKPT_PATH=/mnt/checkpoint-ram/checkpoint.tar
RESULTS_DIR=./results/20251028_224932
NVIDIA_LIBS_PATH=/opt/nvidia-libs
MODELS_CACHE=/models
WARMUP_REQUESTS=5
HEALTH_CHECK_TIMEOUT=300
```

### Podman Container Configuration
```bash
podman run -d \
  --name vllm-llm-demo \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt=label=disable \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  -p 8000:8000 \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code
```

---

## 📚 References

### User's Working Command Sequence
```bash
export CONT_NAME="vllm-llm-demo"
export API_PORT="8000"
export MODEL_ID="Qwen/Qwen2-1.5B-Instruct"

# Create seccomp profile (done in setup.sh)
# Start container (done in create-checkpoint.py)
# Wait for ready (done in create-checkpoint.py)

# Checkpoint
podman container checkpoint --export="$CKPT_PATH" --ignore-volumes "$CONT_NAME"

# Remove container
podman rm -f "$CONT_NAME"

# Restore
time podman container restore --import="$CKPT_PATH" --ignore-volumes
```

### Key Observations from User
- Commands should use **SAME name and path** for checkpoint AND restore
- User questioned **why restore is taking way longer**
- User wants us to **"Ultrathink"** about the problem

---

## 🎯 Next Session Action Items

1. **Get user's actual TFFT numbers** from their working setup
2. **Check for CRIU version updates** or known GPU restore issues
3. **Test checkpoint at different stages** (before full model load)
4. **Investigate CRIU restore flags** for GPU optimization
5. **Compare with direct CRIU** (bypass podman) if needed
6. **Consider alternative approaches** if CRIU GPU restore is fundamentally slow on this system

---

## 💾 Checkpoint Information

**Location**: `/mnt/checkpoint-ram/checkpoint.tar`
**Size**: 3.6 GB
**Storage**: tmpfs (RAM)
**Created**: Oct 28, 2025 22:52 UTC
**Model**: Qwen/Qwen2-1.5B-Instruct
**GPU State**: ~11GB VRAM captured
**Container**: vllm-llm-demo

**Validation**: ✅ Container restores successfully and serves inference

---

## 🔬 Hypothesis for Performance Issue

The fundamental problem appears to be:

1. **GPU memory is the bottleneck**: 11GB of VRAM needs to be restored
2. **CRIU GPU support may be immature**: v4.1.1 might have slow GPU restore
3. **Data path inefficiency**: Checkpoint (RAM) → CPU RAM → GPU VRAM
4. **Copy overhead exceeds model loading**: Restoring GPU state takes longer than loading model from NVMe

**Expected behavior**: Restore from RAM should be 10-30x faster
**Actual behavior**: Restore is 2.1x SLOWER than baseline
**Conclusion**: Something is fundamentally wrong with GPU restore performance on this system

---

## ✅ What Successfully Validated

- ✅ All scripts use PEP 723 format with uv
- ✅ Checkpoint/restore commands match user's working setup exactly
- ✅ Checkpoint is correctly stored in tmpfs (RAM)
- ✅ Container works perfectly after restore
- ✅ Baseline measurement works correctly
- ✅ Infrastructure (podman, CRIU, GPU) all functional

## ❌ What Needs Investigation

- ❌ Why is CRIU restore 2.1x slower than cold start?
- ❌ Is this expected behavior for CRIU v4.1.1 with GPU?
- ❌ What optimization flags are we missing?
- ❌ Should we checkpoint at a different stage?
- ❌ Is there a system-level configuration issue?

---

**Status**: Awaiting user feedback on expected performance and next steps for optimization.
