#!/bin/bash
echo "=== Testing with --ignore-static-mac only ==="
echo ""

podman stop vllm-llm-demo 2>/dev/null && sleep 2

TIMES_BASELINE=()
TIMES_IGNORE=()

echo "Phase 1: Baseline (no flags)"
for i in 1 2 3; do
  START=$(date +%s%N)
  podman container restore --keep vllm-llm-demo 2>&1 >/dev/null
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "  Test $i: ${ELAPSED}s"
  TIMES_BASELINE+=($ELAPSED)
  podman stop vllm-llm-demo 2>/dev/null && sleep 2
done

echo ""
echo "Phase 2: With --ignore-static-mac"
for i in 1 2 3; do
  START=$(date +%s%N)
  podman container restore --keep --ignore-static-mac vllm-llm-demo 2>&1 >/dev/null
  END=$(date +%s%N)
  ELAPSED=$(awk "BEGIN {print ($END - $START) / 1000000000}")
  echo "  Test $i: ${ELAPSED}s"
  TIMES_IGNORE+=($ELAPSED)
  podman stop vllm-llm-demo 2>/dev/null && sleep 2
done

echo ""
echo "=== Results ==="
SUM_BASE=0
for t in "${TIMES_BASELINE[@]}"; do
  SUM_BASE=$(awk "BEGIN {print $SUM_BASE + $t}")
done
AVG_BASE=$(awk "BEGIN {print $SUM_BASE / ${#TIMES_BASELINE[@]}}")

SUM_IGN=0
for t in "${TIMES_IGNORE[@]}"; do
  SUM_IGN=$(awk "BEGIN {print $SUM_IGN + $t}")
done
AVG_IGN=$(awk "BEGIN {print $SUM_IGN / ${#TIMES_IGNORE[@]}}")

echo "Baseline average: ${AVG_BASE}s"
echo "With --ignore-static-mac: ${AVG_IGN}s"
DIFF=$(awk "BEGIN {print $AVG_BASE - $AVG_IGN}")
printf "Difference: %.3fs (%.1f%%)\n" $DIFF $(awk "BEGIN {print ($DIFF / $AVG_BASE) * 100}")
