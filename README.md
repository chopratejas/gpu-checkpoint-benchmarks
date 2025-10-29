# CRIU GPU Cold Start TFFT Benchmark Suite

A comprehensive benchmarking suite for measuring Time-to-First-Token (TFFT) performance comparing traditional cold start vs CRIU GPU checkpoint/restore for vLLM inference workloads.

## Overview

This project aims to achieve **InferX and Modal-level cold start performance** by using CRIU (Checkpoint/Restore In Userspace) with GPU support to drastically reduce model initialization time. Instead of loading model weights from storage into GPU VRAM on every cold start, we checkpoint a fully-initialized vLLM inference server and restore it from DRAM.

### The Problem

Modern LLM inference providers face a challenge: **GPU VRAM is expensive**. When models aren't being used, providers want to swap them out to cheaper storage. Traditional approaches require:

1. Loading multi-GB model weights from NVMe/SSD
2. Copying weights from CPU memory to GPU VRAM
3. Initializing CUDA contexts
4. Setting up vLLM's KV cache and serving infrastructure

This can take **30-60+ seconds** for larger models, resulting in poor user experience.

### The Solution

**CRIU GPU checkpoint/restore** captures the entire process state including:
- GPU VRAM contents (model weights already loaded)
- CUDA context (fully initialized)
- Python runtime (warm and ready)
- vLLM serving stack (ready to serve)

Storing this checkpoint in **DRAM (tmpfs)** enables restoration in **sub-second to single-digit seconds**, dramatically improving cold start TFFT.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Benchmark Suite Flow                         │
└─────────────────────────────────────────────────────────────────┘

1. Setup (setup.sh)
   ├─ Install uv package manager
   ├─ Create tmpfs ramdisk (16GB @ /mnt/checkpoint-ram)
   ├─ Verify CRIU, Podman, NVIDIA GPU
   ├─ Create seccomp profile (disable io_uring)
   └─ Generate .env configuration

2. Create Checkpoint (create-checkpoint.py)
   ├─ Start vLLM container with Qwen/Qwen2-1.5B-Instruct
   ├─ Wait for health check (model fully loaded)
   ├─ Run 5 warmup inferences (CUDA fully initialized)
   ├─ CRIU checkpoint → /mnt/checkpoint-ram/checkpoint.tar
   └─ Log checkpoint size and creation time

3. Benchmark Loop (run-benchmarks.py)
   │
   ├─ Baseline Benchmark (benchmark-baseline.py)
   │  ├─ T0: Start timer
   │  ├─ podman run vllm (cold start)
   │  ├─ T1: Container started
   │  ├─ T2: API health check OK (model loaded)
   │  ├─ T3: First token received (TFFT)
   │  ├─ T4: Full response
   │  └─ Log: container_start, model_load, tfft, total_cold_start_tfft
   │
   ├─ [10s cooldown]
   │
   ├─ CRIU Benchmark (benchmark-criu.py)
   │  ├─ T0: Start timer
   │  ├─ podman restore from checkpoint.tar
   │  ├─ T1: Container restored
   │  ├─ T2: API health check OK
   │  ├─ T3: First token received (TFFT)
   │  ├─ T4: Full response
   │  └─ Log: restore_time, reinit_time, tfft, total_criu_tfft
   │
   └─ [Repeat N iterations]

4. Analysis (analyze-results.py)
   ├─ Load all JSON results from results/
   ├─ Calculate statistics (mean, median, stddev, min, max)
   ├─ Compute speedup ratio (baseline_tfft / criu_tfft)
   ├─ Generate markdown report
   └─ Display rich tables and charts
```

## Requirements

### System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 32GB+ recommended (16GB for tmpfs + model + OS)
- **Storage**: 50GB+ free space for models and checkpoints

### Software Requirements

- **CRIU**: 3.15+ with GPU support
- **Podman**: 4.0+ (rootless mode supported)
- **NVIDIA Drivers**: Latest drivers with CUDA 11.8+
- **Python**: 3.11+ (managed via uv)
- **uv**: Python package manager (auto-installed by setup.sh)

### Python Dependencies

All dependencies are managed via **uv** with PEP 723 inline script metadata:

- `httpx` - HTTP client for API requests
- `rich` - Beautiful terminal UI
- `numpy` - Statistical calculations
- `python-dotenv` - Environment variable management

## Quick Start

### 1. Initial Setup

```bash
cd /root/gpu-load

# Run setup script (requires sudo for tmpfs mount)
sudo bash setup.sh

# Source environment
source .env
source $HOME/.cargo/env  # If uv was just installed
```

### 2. Create Checkpoint

```bash
# Create the golden checkpoint (one-time operation)
./create-checkpoint.py
```

This will:
- Start vLLM container with Qwen/Qwen2-1.5B-Instruct
- Wait for model to fully load (~30-60s)
- Run warmup inferences
- Create checkpoint in /mnt/checkpoint-ram/checkpoint.tar (~4-6GB)

### 3. Run Benchmarks

```bash
# Run 5 iterations of baseline + CRIU benchmarks
./run-benchmarks.py

# Or customize
./run-benchmarks.py --iterations 10
./run-benchmarks.py --baseline-only
./run-benchmarks.py --criu-only
```

### 4. View Results

Results are automatically analyzed and displayed. You can also run analysis manually:

```bash
./analyze-results.py
```

View the detailed report:

```bash
cat results/benchmark_report.md
```

## Detailed Usage

### setup.sh

Environment initialization script.

**What it does:**
- Installs `uv` if not present
- Creates 16GB tmpfs ramdisk at `/mnt/checkpoint-ram`
- Verifies CRIU, Podman, NVIDIA GPU accessibility
- Creates seccomp profile to disable io_uring
- Generates `.env` configuration file

**Usage:**
```bash
sudo bash setup.sh
```

**Output:**
- `.env` - Environment configuration
- `/mnt/checkpoint-ram/` - Mounted tmpfs ramdisk
- `/etc/containers/seccomp.d/no-io-uring.json` - Seccomp profile

### create-checkpoint.py

Creates the golden checkpoint of a fully-initialized vLLM server.

**Configuration (from .env):**
- `MODEL_ID` - HuggingFace model to load
- `GPU_MEMORY_UTIL` - GPU memory utilization (0.90 = 90%)
- `MAX_MODEL_LEN` - Maximum sequence length
- `WARMUP_REQUESTS` - Number of warmup inferences

**Usage:**
```bash
./create-checkpoint.py
```

**Podman Configuration:**
```bash
podman run -d --name vllm-llm-demo \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt=label=disable \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2-1.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code
```

**Output:**
- `/mnt/checkpoint-ram/checkpoint.tar` - CRIU checkpoint (~4-6GB for 1.5B model)

### benchmark-baseline.py

Measures traditional cold start performance.

**Timing Points:**
- **T0**: Before `podman run` starts
- **T1**: Container started
- **T2**: API health check passes (model loaded)
- **T3**: First token received
- **T4**: Full response completed

**Metrics Captured:**
- `container_start_time` (T1-T0): Container startup overhead
- `model_load_time` (T2-T1): Model loading from storage → GPU VRAM
- `inference_tfft` (T3-T2): Time to first token after API ready
- `total_cold_start_tfft` (T3-T0): **PRIMARY METRIC**
- `full_response_time` (T4-T0): Complete end-to-end time

**Usage:**
```bash
./benchmark-baseline.py
```

**Output:**
- `results/TIMESTAMP/baseline_TIMESTAMP.json` - JSON metrics

### benchmark-criu.py

Measures CRIU checkpoint restore performance.

**Timing Points:**
- **T0**: Before `podman restore` starts
- **T1**: Container restored
- **T2**: API health check passes
- **T3**: First token received
- **T4**: Full response completed

**Metrics Captured:**
- `restore_time` (T1-T0): CRIU restore from DRAM
- `reinitialization_time` (T2-T1): Service becoming ready
- `inference_tfft` (T3-T2): Time to first token
- `total_criu_tfft` (T3-T0): **PRIMARY METRIC**
- `full_response_time` (T4-T0): Complete end-to-end time

**Usage:**
```bash
./benchmark-criu.py
```

**Output:**
- `results/TIMESTAMP/criu_TIMESTAMP.json` - JSON metrics

### run-benchmarks.py

Orchestrates the complete benchmark suite.

**Options:**
```bash
./run-benchmarks.py --help

Options:
  --iterations N        Number of benchmark iterations (default: 5)
  --skip-checkpoint     Skip checkpoint creation verification
  --baseline-only       Only run baseline benchmarks
  --criu-only          Only run CRIU benchmarks
```

**Example:**
```bash
# Run 10 iterations with full analysis
./run-benchmarks.py --iterations 10

# Quick baseline-only test
./run-benchmarks.py --baseline-only --iterations 3
```

**Workflow:**
1. Verifies checkpoint exists (creates if missing)
2. Runs N iterations of baseline + CRIU benchmarks
3. Adds 10s cooldown between runs
4. Calls analyze-results.py automatically
5. Displays summary table

### analyze-results.py

Statistical analysis and report generation.

**Calculations:**
- Mean, median, standard deviation
- Min/max values
- Speedup ratio: `baseline_mean / criu_mean`
- Percentage improvement

**Usage:**
```bash
./analyze-results.py
```

**Output:**
- **Console**: Rich tables with color-coded statistics
- **File**: `results/benchmark_report.md` - Markdown report

**Example Output:**
```
╭─────────────────── Summary Comparison ────────────────────╮
│ Metric              Baseline      CRIU        Speedup     │
├────────────────────────────────────────────────────────────┤
│ Total TFFT          47.35s        2.18s       21.7x faster│
│ Container/Restore   1.23s         0.45s       2.7x faster │
│ Model Load/Reinit   45.12s        1.28s       35.3x faster│
│ Inference TFFT      1.00s         0.45s       2.2x faster │
╰────────────────────────────────────────────────────────────╯
```

## Understanding the Metrics

### Baseline (Traditional Cold Start)

**What happens:**
1. Podman starts a new container
2. vLLM process launches
3. Model weights load from `/models` (NVMe) into CPU RAM
4. Model weights copy from CPU RAM to GPU VRAM
5. CUDA initializes
6. vLLM sets up KV cache and serving infrastructure
7. API becomes ready
8. First inference request processes

**Bottlenecks:**
- Large model weight loading (I/O bound)
- CPU → GPU memory transfer (PCIe bandwidth)
- CUDA initialization (GPU driver overhead)
- Python runtime startup

### CRIU Restore (Checkpoint/Restore)

**What happens:**
1. CRIU restores process state from tmpfs (DRAM)
2. GPU VRAM already contains model weights
3. CUDA context already initialized
4. Python runtime already warm
5. vLLM serving stack already set up
6. API becomes ready (minimal reinitialization)
7. First inference request processes

**Advantages:**
- No model weight loading (already in GPU memory)
- No CUDA initialization (context preserved)
- No Python startup (runtime warm)
- Fast restore from DRAM (high bandwidth)

**Expected Speedup:**
- **Small models (1.5B)**: 10-20x faster
- **Medium models (7B)**: 15-30x faster
- **Large models (13B+)**: 20-50x faster

## Test Prompts

The benchmark uses a simple test prompt to measure TFFT:

```python
"Tell me a story in exactly 100 words."
```

This prompt is:
- Short enough to minimize variation
- Long enough to ensure model is working
- Deterministic in expected output length

For production benchmarks, you may want to test with:
- Various prompt lengths
- Different tasks (code generation, question answering, etc.)
- Different sampling parameters (temperature, top_p)

## Extending to Other Models

To test with different models, edit `.env`:

```bash
# Small models
MODEL_ID=Qwen/Qwen2-1.5B-Instruct
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Medium models (requires more RAM/VRAM)
MODEL_ID=meta-llama/Llama-2-7b-chat-hf
MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2

# Large models (requires 40GB+ VRAM)
MODEL_ID=meta-llama/Llama-2-13b-chat-hf
```

**Important**: Larger models require:
- More VRAM (adjust `GPU_MEMORY_UTIL`)
- Larger tmpfs (adjust mount size)
- More time to create checkpoint
- Longer restore times (but still much faster than cold start)

## Troubleshooting

### CRIU Checkpoint Fails

**Error**: `CRIU checkpoint failed with exit code X`

**Solutions:**
1. Check CRIU supports GPU: `criu check --feature mem_dirty_track`
2. Ensure running with sufficient privileges
3. Verify io_uring is disabled (seccomp profile applied)
4. Check podman version: `podman --version` (4.0+ required)

### GPU Not Accessible in Container

**Error**: `Failed to initialize CUDA` or `No GPU found`

**Solutions:**
1. Verify GPU devices exist: `ls -la /dev/nvidia*`
2. Check NVIDIA drivers: `nvidia-smi`
3. Ensure NVIDIA libraries mounted: `ls /opt/nvidia-libs`
4. Try running with `--device nvidia.com/gpu=all` if using CDI

### Out of Memory (OOM)

**Error**: Container killed or system freezes

**Solutions:**
1. Reduce `GPU_MEMORY_UTIL` (e.g., 0.80 instead of 0.90)
2. Reduce `MAX_MODEL_LEN` (e.g., 2048 instead of 4096)
3. Increase system RAM
4. Use a smaller model
5. Increase tmpfs size if checkpoint is large

### Checkpoint Restore Hangs

**Error**: `podman restore` doesn't complete

**Solutions:**
1. Check if container name conflicts: `podman ps -a`
2. Remove stale containers: `podman rm -f vllm-llm-demo`
3. Verify checkpoint integrity: `ls -lh /mnt/checkpoint-ram/`
4. Check CRIU logs: `journalctl -xe | grep criu`

### Health Check Timeout

**Error**: `Health check timed out after 300s`

**Solutions:**
1. Increase `HEALTH_CHECK_TIMEOUT` in `.env`
2. Check container logs: `podman logs vllm-llm-demo`
3. Verify model is downloading: `du -sh /models`
4. Check network connectivity for HuggingFace
5. Pre-download model: `huggingface-cli download MODEL_ID`

## Performance Tips

### Optimize Checkpoint Size

1. **Use quantized models**: GPTQ, AWQ, or GGUF formats
2. **Reduce max_model_len**: Smaller KV cache = smaller checkpoint
3. **Adjust gpu_memory_utilization**: Lower = smaller VRAM footprint

### Optimize Restore Speed

1. **Use tmpfs (DRAM)**: Fastest restore (current setup)
2. **Use NVMe with direct I/O**: If tmpfs not feasible
3. **Increase tmpfs size**: Avoid swapping during restore
4. **Pin tmpfs to NUMA node**: For multi-socket systems

### Optimize Baseline Speed

1. **Pre-download models**: Avoid HuggingFace download time
2. **Use local SSD/NVMe**: Faster than network storage
3. **Enable model quantization**: Less data to load
4. **Warm kernel caches**: Run once to populate page cache

## Project Structure

```
/root/gpu-load/
├── setup.sh                  # Environment setup
├── create-checkpoint.py      # Checkpoint creation
├── benchmark-baseline.py     # Baseline benchmark
├── benchmark-criu.py        # CRIU benchmark
├── run-benchmarks.py        # Orchestration
├── analyze-results.py       # Analysis and reporting
├── utils.py                 # Shared utilities
├── .env                     # Configuration (generated)
├── results/                 # Benchmark results (timestamped)
│   ├── YYYYMMDD_HHMMSS/
│   │   ├── baseline_*.json
│   │   ├── criu_*.json
│   │   └── benchmark_report.md
└── logs/                    # Application logs
```

## Expected Results

For **Qwen/Qwen2-1.5B-Instruct** on a modern GPU (A100, H100, RTX 4090):

| Metric | Baseline | CRIU | Speedup |
|--------|----------|------|---------|
| Total TFFT | 30-50s | 1-3s | **10-30x** |
| Container Start | 1-2s | - | - |
| Restore | - | 0.3-1s | - |
| Model Load | 25-45s | - | - |
| Reinitialization | - | 0.5-2s | **15-30x** |
| Inference TFFT | 0.5-1s | 0.3-0.8s | 1.5-2x |

**Key Insight**: Most speedup comes from **eliminating model loading**. The checkpoint already has model weights in GPU VRAM.

## References

- [CRIU GPU Support](https://criu.org/GPU)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Podman Checkpoint/Restore](https://docs.podman.io/en/latest/markdown/podman-container-checkpoint.1.html)
- [InferX Cold Start Benchmarks](https://www.inferx.com/)
- [Modal Cold Start Performance](https://modal.com/docs/guide/cold-start)

## License

This benchmark suite is provided as-is for research and evaluation purposes.

## Contributing

To contribute improvements:

1. Test with different models and document results
2. Optimize CRIU checkpoint/restore parameters
3. Add support for multi-GPU setups
4. Implement distributed checkpointing
5. Add more comprehensive error handling

## Acknowledgments

- CRIU team for GPU checkpoint support
- vLLM team for the excellent inference engine
- Podman team for container checkpoint integration
- Lambda Labs for GPU cloud infrastructure
