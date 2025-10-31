#!/bin/bash

set -e

echo "=== CRIU Memory Timing Instrumentation Test for vLLM ==="
echo ""

CONTAINER_NAME="vllm-llm-demo"
CHECKPOINT_DIR="/var/lib/containers/storage/overlay-containers"

# Clean up any existing container
echo "Step 1: Cleaning up existing containers..."
podman rm -f "$CONTAINER_NAME" 2>/dev/null || true
sleep 2

# Start fresh vLLM container
echo ""
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

# Wait for health check
echo ""
echo "Step 3: Waiting for container to be healthy..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Container is healthy (after ${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Give it a moment to stabilize
sleep 5

# Create checkpoint
echo ""
echo "Step 4: Creating checkpoint..."
time podman container checkpoint "$CONTAINER_NAME" 2>&1 | tee /tmp/checkpoint-timing.log

# Check checkpoint size
echo ""
echo "Step 5: Checkpoint Info..."
CONTAINER_ID=$(podman inspect "$CONTAINER_NAME" --format '{{.Id}}' 2>/dev/null | head -c 12)
if [ -n "$CONTAINER_ID" ]; then
    CKPT_PATH="$CHECKPOINT_DIR/${CONTAINER_ID}/userdata/checkpoint"
    if [ -d "$CKPT_PATH" ]; then
        echo "Checkpoint location: $CKPT_PATH"
        du -sh "$CKPT_PATH"
        echo "Pages file:"
        find "$CKPT_PATH" -name "pages*.img" -exec ls -lh {} \;
    fi
fi

# Note: After checkpoint, container is in "Exited" state with checkpoint data preserved
# DO NOT use 'podman rm' as it will delete the checkpoint!

# Restore with timing instrumentation
echo ""
echo "Step 6: Container checkpointed (in Exited state, checkpoint preserved)"
echo "========================================="
echo "Step 7: RESTORING WITH TIMING INSTRUMENTATION"
echo "========================================="
echo ""
echo "Watch for 'Memory Restore Timing Summary' in the output..."
echo ""

time podman container restore "$CONTAINER_NAME" 2>&1 | tee /tmp/restore-timing.log

# Wait for container to be healthy after restore
echo ""
echo "Step 8: Verifying restored container..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Restored container is healthy (after ${i}s)"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Extract timing information
echo ""
echo "========================================="
echo "TIMING ANALYSIS"
echo "========================================="
echo ""

# Find the restore.log
CONTAINER_ID=$(podman inspect "$CONTAINER_NAME" --format '{{.Id}}' 2>/dev/null | head -c 12)
if [ -n "$CONTAINER_ID" ]; then
    RESTORE_LOG="$CHECKPOINT_DIR/${CONTAINER_ID}/userdata/restore.log"

    if [ -f "$RESTORE_LOG" ]; then
        echo "Checking restore log: $RESTORE_LOG"
        echo ""

        if grep -q "Memory Restore Timing Summary" "$RESTORE_LOG"; then
            echo "✓ FOUND TIMING INSTRUMENTATION!"
            echo ""
            grep -A 30 "Memory Restore Timing Summary" "$RESTORE_LOG"
        else
            echo "⚠ No timing summary found. Checking for timing data..."
            grep -E "premap_priv_vmas|restore_priv_vma_content|read_pages|CUDA|resume" "$RESTORE_LOG" | head -20
        fi

        echo ""
        echo "Full restore log saved at: $RESTORE_LOG"
    else
        echo "⚠ Restore log not found at: $RESTORE_LOG"
        echo "Checking /tmp/restore-timing.log instead..."
        grep -A 30 "Memory Restore Timing Summary" /tmp/restore-timing.log 2>/dev/null || echo "Not found in podman output either"
    fi
else
    echo "⚠ Could not find container ID"
fi

echo ""
echo "========================================="
echo "TEST COMPLETE!"
echo "========================================="
echo ""
echo "Summary files:"
echo "  - Checkpoint log: /tmp/checkpoint-timing.log"
echo "  - Restore log: /tmp/restore-timing.log"
if [ -n "$RESTORE_LOG" ] && [ -f "$RESTORE_LOG" ]; then
    echo "  - CRIU restore log: $RESTORE_LOG"
fi
