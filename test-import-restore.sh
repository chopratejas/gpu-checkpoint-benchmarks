#!/bin/bash
echo "=== Testing Import-based Restore with Compression ==="
echo "Archive: /mnt/checkpoint-ram/checkpoint-zstd.tar (zstd compressed)"
echo ""

podman rm -f vllm-llm-demo 2>/dev/null

TIMES=()
for i in 1 2 3 4 5; do
  echo "--- Iteration $i ---"
  START=$(date +%s%N)
  podman container restore --import /mnt/checkpoint-ram/checkpoint-zstd.tar 2>&1 | grep -E "(vllm|Restored)" || echo "Import complete"
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "Import+Restore time: ${ELAPSED}s"
  TIMES+=($ELAPSED)
  
  sleep 2
  timeout 5 curl -s http://localhost:8000/health > /dev/null && echo "Health: OK" || echo "Health: TIMEOUT"
  
  podman rm -f vllm-llm-demo 2>/dev/null
  sleep 2
  echo ""
done

echo "=== Summary ==="
SUM=0
for t in "${TIMES[@]}"; do
  SUM=$(awk "BEGIN {print $SUM + $t}")
done
AVG=$(awk "BEGIN {print $SUM / ${#TIMES[@]}}")
echo "Average time: ${AVG}s"
echo "Baseline (in-place restore): 5.40s"
DIFF=$(awk "BEGIN {print $AVG - 5.40}")
echo "Difference: ${DIFF}s"
