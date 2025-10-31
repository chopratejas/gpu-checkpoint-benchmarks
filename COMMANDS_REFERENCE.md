# GPU-Load Checkpoint/Restore Commands Reference

Quick reference for all actual commands executed during checkpoint/restore operations.

---

## CHECKPOINT CREATION

### Full Workflow
```bash
# 1. Remove existing container
podman rm -f vllm-checkpoint

# 2. Start vLLM container with GPU access
podman run -d --name vllm-checkpoint \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --privileged \
  --security-opt=label=disable \
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
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 1024 \
  --enforce-eager \
  --load-format safetensors \
  --trust-remote-code

# 3. Wait for health check (poll every 2 seconds until 200 OK)
curl -s http://localhost:8000/health

# 4. Send warmup inference requests (5 times)
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "stream": true,
    "temperature": 0.7
  }'

# 5. Create checkpoint
podman container checkpoint --print-stats vllm-checkpoint

# 6. Verify checkpoint
CONTAINER_ID=$(podman inspect --format '{{.Id}}' vllm-checkpoint)
ls -lh /var/lib/containers/storage/overlay-containers/$CONTAINER_ID/userdata/checkpoint/
```

### Python Script
```bash
# Run the checkpoint creation script (PEP 723 with uv)
cd /root/gpu-load
./create-checkpoint.py
```

---

## RESTORE OPERATIONS

### Basic Restore (in-place)
```bash
podman container restore vllm-checkpoint
```

### Restore with Keep Flag (keep running)
```bash
podman container restore --keep vllm-checkpoint
```

### Restore with Network Compatibility Flags
```bash
podman container restore --keep \
  --ignore-static-ip \
  --ignore-static-mac \
  vllm-checkpoint
```

### Restore with Custom CRIU Binary
```bash
podman container restore \
  --runtime-opt=runtime_criu_path=/root/criu/criu/criu \
  --keep \
  vllm-checkpoint
```

### Restore from Tar Export
```bash
# First export checkpoint to tar
podman container checkpoint --export /mnt/checkpoint-ram/checkpoint.tar vllm-checkpoint

# Then restore from tar
podman container restore \
  --import /mnt/checkpoint-ram/checkpoint.tar \
  vllm-checkpoint
```

### Compressed Checkpoint Export/Restore
```bash
# Export and compress with zstd
podman container checkpoint --export - vllm-checkpoint | zstd -o checkpoint.tar.zst

# Restore from compressed tar
zstd -d checkpoint.tar.zst -c | podman container restore --import - vllm-checkpoint
```

---

## HEALTH CHECKS & TESTING

### Health Check
```bash
# Simple health check
curl -s http://localhost:8000/health

# With output
curl -v http://localhost:8000/health

# Health check with timeout
timeout 5 curl -s http://localhost:8000/health > /dev/null && echo "OK" || echo "FAIL"
```

### Test Inference
```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "prompt": "Tell me a story",
    "max_tokens": 100,
    "stream": false
  }' | jq .
```

### Stream Inference (with TFFT measurement)
```bash
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-1.5B-Instruct",
    "prompt": "What is AI?",
    "max_tokens": 50,
    "stream": true
  }'
```

---

## CONTAINER MANAGEMENT

### List Containers
```bash
# List all containers
podman ps -a

# List containers with filtering
podman ps -a --filter name=vllm

# Get container ID
podman inspect --format '{{.Id}}' vllm-checkpoint

# Get short container ID
podman inspect --format '{{.Id}}' vllm-checkpoint | cut -c1-12
```

### Container Lifecycle
```bash
# Stop container
podman stop vllm-checkpoint

# Start container (if checkpoint exists)
podman start vllm-checkpoint

# Remove container
podman rm vllm-checkpoint

# Force remove (stop + remove)
podman rm -f vllm-checkpoint
```

### Container Inspection
```bash
# Inspect container details
podman inspect vllm-checkpoint

# Get specific info
podman inspect --format '{{.State.Status}}' vllm-checkpoint
podman inspect --format '{{.NetworkSettings.IPAddress}}' vllm-checkpoint
podman inspect --format '{{.Mounts}}' vllm-checkpoint
```

---

## CHECKPOINT FILE OPERATIONS

### Checkpoint Location
```bash
# Get checkpoint directory
CONTAINER_ID=$(podman inspect --format '{{.Id}}' vllm-checkpoint)
CKPT_DIR="/var/lib/containers/storage/overlay-containers/$CONTAINER_ID/userdata/checkpoint"
echo $CKPT_DIR

# List checkpoint files
ls -lh $CKPT_DIR

# Check checkpoint size
du -sh $CKPT_DIR

# Total size with all files
du -shc $CKPT_DIR/*
```

### Checkpoint Export
```bash
# Export to tar
podman container checkpoint --export /mnt/checkpoint-ram/checkpoint.tar vllm-checkpoint

# Check tar size
du -sh /mnt/checkpoint-ram/checkpoint.tar

# List tar contents
tar -tzf /mnt/checkpoint-ram/checkpoint.tar | head -20

# Extract tar for analysis
tar -xf /mnt/checkpoint-ram/checkpoint.tar -C /tmp/ckpt-analysis
```

### Checkpoint Compression
```bash
# Gzip compression
tar czf /mnt/checkpoint-ram/checkpoint.tar.gz /var/lib/containers/storage/overlay-containers/*/userdata/checkpoint

# Zstd compression
tar -I zstd -cf /mnt/checkpoint-ram/checkpoint.tar.zst /var/lib/containers/storage/overlay-containers/*/userdata/checkpoint

# Compare sizes
du -sh /mnt/checkpoint-ram/checkpoint.tar*
```

---

## PERFORMANCE MEASUREMENT

### Timing with nanoseconds
```bash
# Start timer in nanoseconds
START=$(date +%s%N)
podman container restore --keep vllm-checkpoint
END=$(date +%s%N)

# Calculate elapsed in seconds
ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
echo "Restore time: ${ELAPSED}s"
```

### Full Benchmark Iteration
```bash
#!/bin/bash
# Measure restore + health + inference

# Clean up
podman rm -f vllm-checkpoint 2>/dev/null

# Time restore
START=$(date +%s%N)
podman container restore --keep vllm-checkpoint
RESTORE_END=$(date +%s%N)

# Time health check
sleep 1
curl -s http://localhost:8000/health > /dev/null
HEALTH_END=$(date +%s%N)

# Time inference
START_INF=$(date +%s%N)
curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2-1.5B-Instruct","prompt":"test","max_tokens":10}' > /dev/null
END=$(date +%s%N)

# Calculate timings
RESTORE_TIME=$(awk "BEGIN {print ($RESTORE_END - $START) / 1000000000}")
HEALTH_TIME=$(awk "BEGIN {print ($HEALTH_END - $RESTORE_END) / 1000000000}")
INF_TIME=$(awk "BEGIN {print ($END - $START_INF) / 1000000000}")

echo "Restore: ${RESTORE_TIME}s, Health: ${HEALTH_TIME}s, Inference: ${INF_TIME}s"

# Clean up
podman rm -f vllm-checkpoint
```

---

## RAMDISK/TMPFS OPERATIONS

### Create tmpfs ramdisk
```bash
# Create 16GB tmpfs for checkpoint storage
sudo mkdir -p /mnt/checkpoint-ram
sudo mount -t tmpfs -o size=16G tmpfs /mnt/checkpoint-ram
sudo chmod 777 /mnt/checkpoint-ram

# Verify mount
mount | grep checkpoint-ram
df -h /mnt/checkpoint-ram
```

### Cleanup tmpfs
```bash
# Unmount tmpfs
sudo umount /mnt/checkpoint-ram

# Remove directory
sudo rmdir /mnt/checkpoint-ram
```

### Check tmpfs usage
```bash
df -h /mnt/checkpoint-ram
du -sh /mnt/checkpoint-ram/*
```

---

## CRIU DEBUGGING

### CRIU version
```bash
criu --version

# Custom CRIU
/root/criu/criu/criu --version
```

### CRIU verbose output
```bash
# Via podman (limited)
podman container checkpoint --verbose vllm-checkpoint

# Direct CRIU call with verbosity
criu dump -v4 -D checkpoint-dir --shell-job -t <PID>
```

### Check CRIU capabilities
```bash
criu check
```

---

## BENCHMARK SCRIPTS

### Run full create checkpoint
```bash
cd /root/gpu-load
./create-checkpoint.py
```

### Run restore speed test (5 iterations)
```bash
cd /root/gpu-load
./test-restore-speed.sh
```

### Test with ignore flags
```bash
cd /root/gpu-load
./test-ignore-flags.sh
```

### Test import/restore from tar
```bash
cd /root/gpu-load
./test-import-restore.sh
```

### Run custom CRIU benchmark
```bash
cd /root/gpu-load
./benchmark-custom-criu.py
```

### Run full benchmark suite
```bash
cd /root/gpu-load
./run-benchmarks.py

# With custom iterations
./run-benchmarks.py --iterations 10

# Baseline only
./run-benchmarks.py --baseline-only

# CRIU only
./run-benchmarks.py --criu-only
```

---

## ENVIRONMENT SETUP

### Load configuration
```bash
source /root/gpu-load/.env
echo $MODEL_ID
echo $CONT_NAME
echo $CHECKPOINT_DIR
```

### Check requirements
```bash
# CRIU
which criu && criu --version

# Podman
podman --version

# NVIDIA
nvidia-smi

# Python
python3 --version

# uv
uv --version
```

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

---

## COMMON TROUBLESHOOTING

### Container won't restore
```bash
# Check if checkpoint exists
CONTAINER_ID=$(podman inspect --format '{{.Id}}' vllm-checkpoint)
ls -la /var/lib/containers/storage/overlay-containers/$CONTAINER_ID/userdata/checkpoint/

# Try with verbose
podman container restore --verbose vllm-checkpoint 2>&1

# Check CRIU support
criu check
```

### Port already in use
```bash
# Find process on port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
API_PORT=8001 ./create-checkpoint.py
```

### GPU device not accessible
```bash
# Check GPU devices
ls -la /dev/nvidia*

# Verify NVIDIA driver
nvidia-smi

# Check container GPU access
podman run --rm --device /dev/nvidia0 docker.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
```

### Health check timeout
```bash
# Check container logs
podman logs vllm-checkpoint

# Increase timeout in .env
HEALTH_CHECK_TIMEOUT=600

# Manually check API
curl -v http://localhost:8000/health
```

### Out of memory
```bash
# Check available memory
free -h

# Check GPU memory
nvidia-smi

# Reduce GPU memory utilization in config
GPU_MEMORY_UTIL=0.50  # Instead of 0.90
```

