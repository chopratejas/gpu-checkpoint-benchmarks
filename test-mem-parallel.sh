#!/bin/bash
echo "=== Testing with mem_parallel Integration ==="
echo ""
echo "CRIU Version:"
criu --version
echo ""

TIMES=()
for i in 1 2 3 4 5 6 7 8; do
  echo "--- Iteration $i ---"
  START=$(date +%s%N)
  podman container restore --keep vllm-llm-demo 2>&1 >/dev/null
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

echo "=== Performance Summary ==="
SUM=0
for t in "${TIMES[@]}"; do
  SUM=$(awk "BEGIN {print $SUM + $t}")
done
AVG=$(awk "BEGIN {print $SUM / ${#TIMES[@]}}")
echo ""
echo "Average restore time: ${AVG}s"
echo "Baseline (before any changes): 5.688s"
IMPROVEMENT=$(awk "BEGIN {print (5.688 - $AVG) * 1000}")
PERCENT=$(awk "BEGIN {print ((5.688 - $AVG) / 5.688) * 100}")
printf "Improvement: %.0fms (%.1f%%)\n" $IMPROVEMENT $PERCENT
