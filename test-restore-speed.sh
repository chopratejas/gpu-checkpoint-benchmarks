#!/bin/bash
echo "=== Testing Restore Speed with Context Pool ==="
echo ""
echo "CRIU Version:"
criu --version
echo ""

for i in 1 2 3 4 5; do
  echo "--- Iteration $i ---"
  START=$(date +%s%N)
  podman container restore --keep vllm-llm-demo 2>&1 | grep -E "(Restored|Error)" || echo "Restore completed"
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "Restore time: ${ELAPSED}s"
  
  sleep 2
  timeout 5 curl -s http://localhost:8000/health > /dev/null && echo "Health: OK" || echo "Health: TIMEOUT"
  
  podman stop vllm-llm-demo 2>/dev/null
  sleep 2
  echo ""
done

echo "=== Test Complete! ==="
