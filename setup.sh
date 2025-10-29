#!/bin/bash

################################################################################
# CRIU GPU Benchmark Suite - Setup Script
#
# This script sets up the environment for running GPU checkpoint/restore
# benchmarks with CRIU, Podman, and vLLM.
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handler
error_exit() {
    log_error "$1"
    exit 1
}

################################################################################
# 1. Check for and install uv (Python package manager)
################################################################################
install_uv() {
    log_info "Checking for uv Python package manager..."

    if command -v uv &> /dev/null; then
        log_success "uv is already installed ($(uv --version))"
    else
        log_info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || error_exit "Failed to install uv"

        # Source the environment to make uv available in current session
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi

        # Verify installation
        if command -v uv &> /dev/null; then
            log_success "uv installed successfully ($(uv --version))"
        else
            log_warning "uv installed but not in PATH. You may need to restart your shell or run: source \$HOME/.cargo/env"
        fi
    fi
}

################################################################################
# 2. Create tmpfs ramdisk for checkpoint storage
################################################################################
create_tmpfs_ramdisk() {
    log_info "Setting up tmpfs ramdisk at /mnt/checkpoint-ram..."

    # Check if directory exists
    if [ ! -d "/mnt/checkpoint-ram" ]; then
        log_info "Creating /mnt/checkpoint-ram directory..."
        sudo mkdir -p /mnt/checkpoint-ram || error_exit "Failed to create /mnt/checkpoint-ram directory"
    fi

    # Check if already mounted
    if mountpoint -q /mnt/checkpoint-ram; then
        log_success "/mnt/checkpoint-ram is already mounted as tmpfs"
        df -h /mnt/checkpoint-ram | tail -1
    else
        log_info "Mounting tmpfs ramdisk (16GB)..."
        sudo mount -t tmpfs -o size=16G tmpfs /mnt/checkpoint-ram || error_exit "Failed to mount tmpfs ramdisk"
        log_success "tmpfs ramdisk mounted successfully"
        df -h /mnt/checkpoint-ram | tail -1
    fi

    # Set appropriate permissions
    sudo chmod 755 /mnt/checkpoint-ram
}

################################################################################
# 3. Verify system requirements
################################################################################
verify_requirements() {
    log_info "Verifying system requirements..."

    local missing_deps=()

    # Check for CRIU
    if ! command -v criu &> /dev/null; then
        missing_deps+=("criu")
        log_error "CRIU is not installed"
    else
        log_success "CRIU found: $(criu --version 2>&1 | head -1)"
    fi

    # Check for Podman
    if ! command -v podman &> /dev/null; then
        missing_deps+=("podman")
        log_error "Podman is not installed"
    else
        log_success "Podman found: $(podman --version)"
    fi

    # Check for nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        missing_deps+=("nvidia-smi")
        log_error "nvidia-smi is not installed (NVIDIA drivers required)"
    else
        log_success "nvidia-smi found: $(nvidia-smi --version | head -1)"
    fi

    # Check for GPU devices
    if [ ! -e /dev/nvidia0 ]; then
        log_error "No GPU device found at /dev/nvidia0"
        missing_deps+=("GPU device")
    else
        log_success "GPU device found at /dev/nvidia0"

        # Try to get GPU info
        if command -v nvidia-smi &> /dev/null; then
            log_info "GPU Information:"
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || log_warning "Could not query GPU information"
        fi
    fi

    # Exit if there are missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        error_exit "Missing required dependencies: ${missing_deps[*]}"
    fi

    log_success "All system requirements verified"
}

################################################################################
# 4. Create seccomp profile to disable io_uring
################################################################################
create_seccomp_profile() {
    log_info "Creating seccomp profile to disable io_uring..."

    local seccomp_dir="/etc/containers/seccomp.d"
    local seccomp_file="${seccomp_dir}/no-io-uring.json"

    # Create directory if it doesn't exist
    if [ ! -d "$seccomp_dir" ]; then
        log_info "Creating seccomp directory: $seccomp_dir"
        sudo mkdir -p "$seccomp_dir" || error_exit "Failed to create seccomp directory"
    fi

    # Create the seccomp profile
    log_info "Writing seccomp profile to $seccomp_file"
    sudo tee "$seccomp_file" > /dev/null <<'EOF'
{
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [
    {
      "names": [
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register"
      ],
      "action": "SCMP_ACT_ERRNO"
    }
  ]
}
EOF

    if [ $? -eq 0 ]; then
        log_success "Seccomp profile created at $seccomp_file"
    else
        error_exit "Failed to create seccomp profile"
    fi
}

################################################################################
# 5. Create directory structure
################################################################################
create_directory_structure() {
    log_info "Creating directory structure..."

    local dirs=("results" "logs")

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir" || error_exit "Failed to create directory: $dir"
            log_success "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
}

################################################################################
# 6. Export and save environment variables
################################################################################
create_env_file() {
    log_info "Creating .env file with environment variables..."

    local env_file=".env"
    local timestamp=$(date +%Y%m%d_%H%M%S)

    cat > "$env_file" <<EOF
# CRIU GPU Benchmark Suite - Environment Variables
# Generated on $(date)

# Model configuration
MODEL_ID=Qwen/Qwen2-1.5B-Instruct
MAX_MODEL_LEN=4096
GPU_MEMORY_UTIL=0.90

# Container configuration
CONT_NAME=vllm-llm-demo
VLLM_IMAGE=docker.io/vllm/vllm-openai:latest
API_PORT=8000

# Checkpoint configuration
CHECKPOINT_DIR=/mnt/checkpoint-ram
CKPT_PATH=/mnt/checkpoint-ram/checkpoint.tar

# Directory paths
RESULTS_DIR=./results/${timestamp}
NVIDIA_LIBS_PATH=/opt/nvidia-libs
MODELS_CACHE=/models

# Performance configuration
WARMUP_REQUESTS=5
HEALTH_CHECK_TIMEOUT=300
EOF

    if [ $? -eq 0 ]; then
        log_success "Environment file created at $env_file"
        log_info "Environment variables preview:"
        cat "$env_file" | grep -v "^#" | grep -v "^$"
    else
        error_exit "Failed to create .env file"
    fi
}

################################################################################
# Main execution
################################################################################
main() {
    log_info "Starting CRIU GPU Benchmark Suite Setup"
    echo "========================================================================"

    # Execute setup steps
    install_uv
    echo ""

    create_tmpfs_ramdisk
    echo ""

    verify_requirements
    echo ""

    create_seccomp_profile
    echo ""

    create_directory_structure
    echo ""

    create_env_file
    echo ""

    # Final summary
    echo "========================================================================"
    log_success "Setup completed successfully!"
    echo ""
    log_info "Next steps:"
    echo "  1. Source the environment file: source .env"
    echo "  2. If uv was just installed: source \$HOME/.cargo/env"
    echo "  3. Review the configuration in .env file"
    echo "  4. Run your benchmark scripts"
    echo ""
    log_info "To persist tmpfs mount across reboots, add to /etc/fstab:"
    echo "  tmpfs  /mnt/checkpoint-ram  tmpfs  size=16G  0  0"
    echo "========================================================================"
}

# Run main function
main "$@"
