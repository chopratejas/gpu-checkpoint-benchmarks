# Quick Start Guide - CRIU GPU Cold Start TFFT Benchmark

## 🚀 Run in 3 Commands

```bash
# 1. Setup environment (installs uv, creates tmpfs, verifies requirements)
sudo bash setup.sh

# 2. Source environment variables
source .env && source $HOME/.cargo/env

# 3. Run complete benchmark suite (creates checkpoint + runs benchmarks)
./run-benchmarks.py
```

That's it! Results will be displayed automatically with speedup metrics.

---

## 📊 What Gets Measured

### Baseline (Traditional Cold Start)
- Start container from scratch
- Load model from NVMe → CPU → GPU VRAM
- Measure total time to first token

### CRIU (Checkpoint/Restore)
- Restore container from checkpoint in DRAM
- Model already in GPU memory
- Measure total time to first token

### Expected Results
**10-30x speedup** for small models (1.5B parameters)

---

## 🔧 Manual Control (Optional)

If you want more control:

```bash
# 1. Setup
sudo bash setup.sh
source .env && source $HOME/.cargo/env

# 2. Create checkpoint once
./create-checkpoint.py

# 3. Run specific benchmarks
./benchmark-baseline.py    # Traditional cold start
./benchmark-criu.py        # CRIU restore

# 4. Analyze results
./analyze-results.py
```

---

## 📈 Customization

Edit `.env` file to change:

```bash
# Try different models
MODEL_ID=Qwen/Qwen2-1.5B-Instruct      # Small (current)
MODEL_ID=meta-llama/Llama-2-7b-chat-hf  # Medium
MODEL_ID=mistralai/Mistral-7B-v0.1      # Medium

# Adjust iterations
./run-benchmarks.py --iterations 10

# Run only one type
./run-benchmarks.py --baseline-only
./run-benchmarks.py --criu-only
```

---

## 📁 Output

Results saved to:
- `results/TIMESTAMP/baseline_*.json` - Baseline metrics
- `results/TIMESTAMP/criu_*.json` - CRIU metrics
- `results/benchmark_report.md` - Detailed analysis

---

## 🐛 Common Issues

**"CRIU not found"**
```bash
sudo apt-get update && sudo apt-get install -y criu
```

**"GPU not accessible"**
```bash
# Check GPU
nvidia-smi

# Verify devices
ls -la /dev/nvidia*
```

**"Checkpoint too large for tmpfs"**
```bash
# Increase tmpfs size (adjust from 16G to 32G)
sudo umount /mnt/checkpoint-ram
sudo mount -t tmpfs -o size=32G tmpfs /mnt/checkpoint-ram
```

---

## 📖 Full Documentation

See `README.md` for complete documentation including:
- Architecture deep-dive
- Metric explanations
- Performance optimization tips
- Troubleshooting guide
- Multi-model testing

---

## 🎯 Goal

Achieve **InferX/Modal-level cold start performance** by:
1. Checkpointing fully-initialized vLLM inference server
2. Storing checkpoint in DRAM (tmpfs)
3. Restoring in sub-second to single-digit seconds
4. Eliminating 30-60s model loading overhead

**Current target**: Sub-3s TFFT for Qwen2-1.5B on modern GPUs
