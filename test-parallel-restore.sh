#!/bin/bash
echo "=== Checkpoint and Parallel Restore Test ==="
echo ""
echo "1. Checkpointing container..."
time podman container checkpoint vllm-llm-demo
echo ""
echo "2. Testing restore with LD_PRELOAD parallel streams..."
echo ""

for i in 1 2 3; do
  echo "--- Iteration $i ---"
  START=$(date +%s.%N)
  podman container restore --keep vllm-llm-demo 2>&1 | tail -1
  END=$(date +%s.%N)
  ELAPSED=$(echo "$END - $START" | bc)
  echo "Restore time: ${ELAPSED}s"
  sleep 1
  curl -s http://localhost:8000/health > /dev/null && echo "Health: OK"
  podman stop vllm-llm-demo 2>/dev/null
  sleep 2
  echo ""
done

echo "=== Test Complete! ==="
