#!/bin/bash
echo "=== Testing Restore with --ignore-static-ip --ignore-static-mac ==="
echo ""

TIMES=()
for i in 1 2 3 4 5; do
  echo "--- Iteration $i ---"
  START=$(date +%s%N)
  podman container restore --keep --ignore-static-ip --ignore-static-mac vllm-llm-demo 2>&1 >/dev/null
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "Restore time: ${ELAPSED}s"
  TIMES+=($ELAPSED)
  
  sleep 2
  timeout 5 curl -s http://localhost:8000/health > /dev/null && echo "Health: OK" || echo "Health: TIMEOUT"
  
  podman stop vllm-llm-demo 2>/dev/null
  sleep 2
  echo ""
done

echo "=== Summary ==="
SUM=0
for t in "${TIMES[@]}"; do
  SUM=$(awk "BEGIN {print $SUM + $t}")
done
AVG=$(awk "BEGIN {print $SUM / ${#TIMES[@]}}")
echo "Average time with --ignore flags: ${AVG}s"
echo "Baseline (no flags): 5.40s"
DIFF=$(awk "BEGIN {print 5.40 - $AVG}")
if (( $(awk "BEGIN {print ($DIFF > 0)}") )); then
  printf "✓ Improvement: %.3fs faster\n" $DIFF
else
  printf "✗ Slower by: %.3fs\n" $(awk "BEGIN {print -1 * $DIFF}")
fi
