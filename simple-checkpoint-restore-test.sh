#!/bin/bash

set -e

CONTAINER_NAME="vllm-llm-demo"

echo "=== Simple Checkpoint/Restore Test with Timing Instrumentation ==="
echo ""

# Clean up
echo "Step 1: Clean up..."
podman rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

# Start container
echo "Step 2: Starting vLLM container..."
podman run -d \
  --name "$CONTAINER_NAME" \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  docker.io/vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.3

# Wait for health
echo "Step 3: Waiting for container health..."
for i in {1..90}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Healthy after ${i}s"
        break
    fi
    sleep 1
done

# Stabilize
sleep 5

# Checkpoint (basic, no export)
echo "Step 4: Creating checkpoint..."
podman container checkpoint "$CONTAINER_NAME"

echo "Step 5: Checkpoint created, container in Exited state"
podman ps -a | grep vllm

# Find checkpoint location
CONTAINER_ID=$(podman inspect "$CONTAINER_NAME" --format '{{.Id}}' | head -c 64)
CKPT_DIR="/var/lib/containers/storage/overlay-containers/${CONTAINER_ID}/userdata/checkpoint"

echo "Step 6: Checkpoint location and size:"
if [ -d "$CKPT_DIR" ]; then
    du -sh "$CKPT_DIR"
    ls "$CKPT_DIR"/pages*.img 2>/dev/null | head -3
fi

# Restore with instrumented CRIU
echo ""
echo "========================================="
echo "Step 7: RESTORING (Watch for timing!)"
echo "========================================="
time podman container restore "$CONTAINER_NAME"

# Wait for health
echo "Step 8: Verifying..."
for i in {1..30}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Healthy after restore (${i}s)"
        break
    fi
    sleep 1
done

# Extract timing from CRIU logs
echo ""
echo "========================================="
echo "TIMING RESULTS"
echo "========================================="

RESTORE_LOG="/var/lib/containers/storage/overlay-containers/${CONTAINER_ID}/userdata/restore.log"

if [ -f "$RESTORE_LOG" ]; then
    echo "Restore log: $RESTORE_LOG"
    echo ""

    if grep -q "Memory Restore Timing Summary" "$RESTORE_LOG"; then
        echo "✓✓✓ SUCCESS! Found timing instrumentation! ✓✓✓"
        echo ""
        grep -A 25 "Memory Restore Timing Summary" "$RESTORE_LOG"
    else
        echo "Checking for any timing data..."
        grep -E "premap|restore_priv|read_pages|memcpy|CUDA|resume" "$RESTORE_LOG" | head -20
    fi
else
    echo "⚠ Restore log not found at: $RESTORE_LOG"
fi

echo ""
echo "=== DONE ==="
