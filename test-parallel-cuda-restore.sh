#!/bin/bash
#
# Test CUDA Parallel Restore with LD_PRELOAD Hooks
# This script tests the newly compiled CRIU with parallel GPU memory restore
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRIU_BIN="/usr/local/sbin/criu"
DRIVER_HOOK="/usr/local/lib/criu/cuda_driver_hook.so"
RUNTIME_HOOK="/usr/local/lib/criu/cuda_preload_hook.so"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [[ ! -f "$CRIU_BIN" ]]; then
        log_error "CRIU binary not found at $CRIU_BIN"
        exit 1
    fi

    if [[ ! -f "$DRIVER_HOOK" ]]; then
        log_error "Driver hook not found at $DRIVER_HOOK"
        exit 1
    fi

    if [[ ! -f "$RUNTIME_HOOK" ]]; then
        log_error "Runtime hook not found at $RUNTIME_HOOK"
        exit 1
    fi

    log_info "CRIU version: $($CRIU_BIN --version | head -1)"
    log_info "Driver hook: $DRIVER_HOOK ($(stat -c%s $DRIVER_HOOK) bytes)"
    log_info "Runtime hook: $RUNTIME_HOOK ($(stat -c%s $RUNTIME_HOOK) bytes)"

    log_success "All prerequisites OK"
}

# Find running container
find_container() {
    log_info "Looking for running vLLM container..."

    CONTAINER_ID=$(podman ps --filter "ancestor=docker.io/vllm/vllm-openai" --format "{{.ID}}" | head -1)

    if [[ -z "$CONTAINER_ID" ]]; then
        log_warn "No running vLLM container found"
        log_info "You can start one with: podman run --rm --privileged --device /dev/nvidia0 ..."
        return 1
    fi

    CONTAINER_NAME=$(podman ps --filter "id=$CONTAINER_ID" --format "{{.Names}}")
    log_success "Found container: $CONTAINER_NAME (ID: $CONTAINER_ID)"

    return 0
}

# Test restore without hooks (baseline)
test_baseline_restore() {
    log_info "=========================================="
    log_info "TEST 1: Baseline Restore (No LD_PRELOAD)"
    log_info "=========================================="

    log_info "Creating checkpoint..."
    podman container checkpoint --print-stats "$CONTAINER_NAME" 2>&1 | tee /tmp/checkpoint_baseline.log

    log_info "Waiting 2 seconds..."
    sleep 2

    log_info "Restoring (baseline - no parallel GPU)..."
    time podman container restore --print-stats "$CONTAINER_NAME" 2>&1 | tee /tmp/restore_baseline.log

    log_info "Checking health..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Container healthy after ${i}s"
            break
        fi
        sleep 1
    done

    log_success "Baseline restore complete"
    echo ""
}

# Test restore with LD_PRELOAD hooks
test_parallel_restore() {
    log_info "================================================"
    log_info "TEST 2: Parallel Restore (WITH LD_PRELOAD Hooks)"
    log_info "================================================"

    # Set environment variables for parallel restore
    export CRIU_CUDA_PARALLEL_RESTORE=1
    export CRIU_CUDA_STREAMS=8
    export CRIU_CUDA_CHUNK_MB=128
    export CRIU_CUDA_USE_PINNED_MEM=1
    export CRIU_CUDA_VERBOSE=1

    log_info "Configuration:"
    log_info "  CRIU_CUDA_PARALLEL_RESTORE=$CRIU_CUDA_PARALLEL_RESTORE"
    log_info "  CRIU_CUDA_STREAMS=$CRIU_CUDA_STREAMS"
    log_info "  CRIU_CUDA_CHUNK_MB=$CRIU_CUDA_CHUNK_MB"
    log_info "  CRIU_CUDA_USE_PINNED_MEM=$CRIU_CUDA_USE_PINNED_MEM"
    log_info "  LD_PRELOAD=$DRIVER_HOOK"

    log_info "Creating checkpoint..."
    podman container checkpoint --print-stats "$CONTAINER_NAME" 2>&1 | tee /tmp/checkpoint_parallel.log

    log_info "Waiting 2 seconds..."
    sleep 2

    log_info "Restoring (with parallel GPU transfer)..."
    LD_PRELOAD="$DRIVER_HOOK" \
        time podman container restore --print-stats "$CONTAINER_NAME" 2>&1 | tee /tmp/restore_parallel.log

    log_info "Checking health..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log_success "Container healthy after ${i}s"
            break
        fi
        sleep 1
    done

    # Look for evidence of parallel restore in logs
    if grep -q "CRIU-CUDA-DRIVER-HOOK" /tmp/restore_parallel.log; then
        log_success "LD_PRELOAD hook WAS ACTIVATED!"
        log_info "Hook messages found in restore log"
    else
        log_warn "LD_PRELOAD hook messages NOT found in log"
        log_warn "This may mean cuda-checkpoint didn't trigger the hooks"
    fi

    log_success "Parallel restore complete"
    echo ""
}

# Compare results
compare_results() {
    log_info "==============================="
    log_info "PERFORMANCE COMPARISON"
    log_info "==============================="

    if [[ -f /tmp/restore_baseline.log ]] && [[ -f /tmp/restore_parallel.log ]]; then
        log_info "Baseline restore log: /tmp/restore_baseline.log"
        log_info "Parallel restore log: /tmp/restore_parallel.log"

        log_info ""
        log_info "Check logs for timing differences"
        log_info "Look for GPU memory transfer times"
    fi
}

# Verify inference works
test_inference() {
    log_info "==============================="
    log_info "TESTING INFERENCE"
    log_info "==============================="

    log_info "Sending test request to vLLM..."

    RESPONSE=$(curl -s http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen/Qwen2-1.5B-Instruct",
            "prompt": "What is CRIU?",
            "max_tokens": 50,
            "temperature": 0.7
        }')

    if echo "$RESPONSE" | jq -e '.choices[0].text' > /dev/null 2>&1; then
        log_success "Inference working!"
        log_info "Response: $(echo "$RESPONSE" | jq -r '.choices[0].text' | head -c 100)..."
    else
        log_error "Inference failed"
        log_error "Response: $RESPONSE"
        return 1
    fi
}

# Main execution
main() {
    log_info "CUDA Parallel Restore Test Suite"
    log_info "=================================="
    echo ""

    check_prerequisites
    echo ""

    if ! find_container; then
        log_error "Cannot proceed without running container"
        exit 1
    fi
    echo ""

    # Run tests
    test_baseline_restore
    sleep 3

    test_parallel_restore
    sleep 3

    compare_results
    echo ""

    test_inference
    echo ""

    log_success "All tests complete!"
    log_info "Check logs at:"
    log_info "  - /tmp/checkpoint_baseline.log"
    log_info "  - /tmp/restore_baseline.log"
    log_info "  - /tmp/checkpoint_parallel.log"
    log_info "  - /tmp/restore_parallel.log"
}

main "$@"
