#!/bin/bash
# Test VMM-enabled CRIU restore for vLLM container
#
# This script demonstrates how to use the VMM lazy restore feature
# with podman container checkpoint/restore

set -euo pipefail

CONTAINER_NAME="${1:-vllm-llm-demo}"
EAGER_MB="${2:-100}"  # Default: Load first 100MB eagerly
WORKERS="${3:-4}"     # Default: 4 parallel workers

echo "========================================="
echo "  VMM Lazy Restore - Container Test"
echo "========================================="
echo "Container: $CONTAINER_NAME"
echo "Eager size: $EAGER_MB MB"
echo "Workers: $WORKERS"
echo "========================================="
echo ""

# Check if container exists
if ! podman ps -a --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
    echo "ERROR: Container '$CONTAINER_NAME' not found"
    echo "Available containers:"
    podman ps -a --format "table {{.Names}}\t{{.Status}}"
    exit 1
fi

# Check if container has checkpoint
CONTAINER_STATE=$(podman inspect "$CONTAINER_NAME" --format "{{.State.Status}}")
if [ "$CONTAINER_STATE" != "exited" ]; then
    echo "WARNING: Container is not in 'exited' state (current: $CONTAINER_STATE)"
    echo "For checkpoint/restore, container should be checkpointed first"
    exit 1
fi

# Enable VMM lazy restore
export CRIU_CUDA_VMM_LAZY=1
export CRIU_CUDA_VMM_EAGER_MB=$EAGER_MB
export CRIU_CUDA_VMM_WORKERS=$WORKERS

echo "VMM Configuration:"
echo "  CRIU_CUDA_VMM_LAZY=$CRIU_CUDA_VMM_LAZY"
echo "  CRIU_CUDA_VMM_EAGER_MB=$CRIU_CUDA_VMM_EAGER_MB"
echo "  CRIU_CUDA_VMM_WORKERS=$CRIU_CUDA_VMM_WORKERS"
echo ""

# Restore container with VMM
echo "Starting VMM lazy restore..."
START_TIME=$(date +%s.%N)

podman container restore --keep "$CONTAINER_NAME" 2>&1 | tee /tmp/vmm-container-restore.log

RESTORE_EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s.%N)
ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo ""
echo "========================================="
echo "  Restore Complete"
echo "========================================="
echo "Exit code: $RESTORE_EXIT_CODE"
echo "Total time: ${ELAPSED}s"
echo "Log saved to: /tmp/vmm-container-restore.log"
echo ""

if [ $RESTORE_EXIT_CODE -eq 0 ]; then
    # Check if container is running
    if podman ps --filter "name=^${CONTAINER_NAME}$" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
        echo "✓ Container is running"

        # Try to hit health endpoint
        if command -v curl &> /dev/null; then
            echo "Checking health endpoint..."
            sleep 2
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "✓ Health endpoint responding"
            else
                echo "⚠ Health endpoint not responding (may need more time)"
            fi
        fi
    else
        echo "⚠ Container restored but not running"
    fi

    # Extract VMM metrics from log
    echo ""
    echo "========================================="
    echo "  VMM Performance Metrics"
    echo "========================================="
    grep -E "TTFT|Lazy restore|bandwidth|VMM.*Complete" /tmp/vmm-container-restore.log || echo "(No VMM metrics found in log)"
    echo ""
else
    echo "✗ Restore failed (exit code: $RESTORE_EXIT_CODE)"
    echo "Check log for errors: /tmp/vmm-container-restore.log"
    exit $RESTORE_EXIT_CODE
fi
