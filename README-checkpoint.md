# vLLM Checkpoint Creation Script

## Overview

The `create-checkpoint.py` script creates a checkpoint of a running vLLM container after performing warmup inference requests. This checkpoint can be used for fast container restoration using CRIU (Checkpoint/Restore In Userspace).

## Files Created

- `/root/gpu-load/create-checkpoint.py` - Main checkpoint creation script (PEP 723 format)
- `/root/gpu-load/utils.py` - Shared utility functions (PEP 723 format)
- `/root/gpu-load/.env` - Environment configuration file
- `/root/gpu-load/.env.example` - Example configuration template

## Script Features

1. **PEP 723 Format**: Uses inline script metadata for dependency management with `uv`
2. **Configuration Management**: Loads settings from `.env` file
3. **Container Management**: Removes existing containers, starts fresh vLLM container
4. **Health Monitoring**: Waits for vLLM API to be ready
5. **Warmup Requests**: Sends 5 inference requests to warm up the model
6. **Checkpoint Creation**: Creates a container checkpoint using podman
7. **Verification**: Checks checkpoint file exists and logs size
8. **Error Handling**: Comprehensive error handling and logging
9. **Timing Measurements**: Tracks time for all operations

## Configuration (.env)

```bash
CONT_NAME=vllm-checkpoint              # Container name
MODEL_ID=meta-llama/Llama-3.2-1B-Instruct  # Model to load
API_PORT=8000                          # API port
CKPT_PATH=/tmp/vllm-checkpoint.tar.gz  # Checkpoint output path
```

## Usage

### Prerequisites

- `uv` installed (for PEP 723 script execution)
- `podman` with checkpoint support
- NVIDIA GPU and drivers configured
- Model files in `/models` directory

### Running the Script

```bash
# Make sure .env file is configured
cd /root/gpu-load

# Run with uv (automatically installs dependencies)
./create-checkpoint.py

# Or explicitly with uv
uv run create-checkpoint.py
```

## Script Workflow

### Step 1: Load Configuration
- Reads `.env` file using `utils.load_env()`
- Validates required environment variables
- Sets defaults for missing values

### Step 2: Remove Existing Container
- Checks for existing containers with same name
- Force removes if found (stops and removes)

### Step 3: Start vLLM Container
Starts podman container with exact configuration:
```bash
podman run -d --name "$CONT_NAME" \
  --device /dev/null:/dev/null:rwm \
  --privileged \
  --security-opt=label=disable \
  --security-opt seccomp=/etc/containers/seccomp.d/no-io-uring.json \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --shm-size 8g \
  -e LD_LIBRARY_PATH=/opt/nvidia-libs:$LD_LIBRARY_PATH \
  -e ASYNCIO_DEFAULT_BACKEND=select \
  -e PYTHON_ASYNCIO_NO_IO_URING=1 \
  -v /opt/nvidia-libs:/opt/nvidia-libs:ro \
  -v /models:/root/.cache/huggingface \
  docker.io/vllm/vllm-openai:latest \
  --model "$MODEL_ID" \
  --host 0.0.0.0 --port ${API_PORT} \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \
  --trust-remote-code
```

### Step 4: Wait for Health Check
- Uses `utils.wait_for_health()` to poll `/health` endpoint
- Timeout: 300 seconds (5 minutes)
- Polls every 2 seconds

### Step 5: Run Warmup Requests
- Sends 5 inference requests using `utils.send_inference()`
- Uses variety of prompts
- Measures TFFT (Time to First Token) and total time
- Max tokens: 50 per request

### Step 6: Create Checkpoint
Creates checkpoint using podman:
```bash
podman container checkpoint \
  --export="$CKPT_PATH" \
  --ignore-volumes \
  "$CONT_NAME"
```

### Step 7: Verify Checkpoint
- Checks checkpoint file exists
- Logs file size (MB/GB)
- Reports creation time

### Step 8: Cleanup
- Removes the container: `podman rm -f "$CONT_NAME"`
- Container is checkpointed, so it can be restored later

## Dependencies

The script uses PEP 723 format with these dependencies:
- `httpx` - HTTP client for API requests
- `rich` - Terminal formatting and progress bars

Dependencies are automatically installed by `uv` when running the script.

## Output Example

```
vLLM Checkpoint Creation Script

Step 1: Loading environment configuration
Loaded 4 environment variables from .env
Configuration loaded:
  Container name: vllm-checkpoint
  API port: 8000
  Checkpoint path: /tmp/vllm-checkpoint.tar.gz

Step 2: Removing any existing containers
Checking for existing container: vllm-checkpoint
No existing container found

Step 3: Starting vLLM container
Starting vLLM container: vllm-checkpoint
Model: meta-llama/Llama-3.2-1B-Instruct
Port: 8000
Container started: a1b2c3d4e5f6

Step 4: Waiting for health check
Waiting for vLLM health check on port 8000...
✓ vLLM is healthy on port 8000 (took 45.23s)
Health check passed in 45.23s

Step 5: Running warmup inference requests
Running 5 warmup inference requests...
Request 1/5 completed
  TFFT: 0.234s, Total: 2.156s, Tokens: 50
...

Step 6: Creating checkpoint
Creating checkpoint: /tmp/vllm-checkpoint.tar.gz
Checkpoint created in 8.45s

Step 7: Verifying checkpoint
Checkpoint verified: /tmp/vllm-checkpoint.tar.gz
Checkpoint size: 1.23 GB

Step 8: Cleaning up container
Container vllm-checkpoint removed

Checkpoint creation completed successfully!
Total time: 67.89s
Checkpoint file: /tmp/vllm-checkpoint.tar.gz
```

## Error Handling

The script includes comprehensive error handling:
- Configuration validation
- Health check timeouts
- Inference request failures
- Checkpoint creation failures
- File verification errors

All errors are logged with rich formatting and the script exits with code 1 on failure.

## Timing Measurements

The script tracks timing for:
- Health check duration
- Individual inference requests (TFFT, total time, tokens/sec)
- Checkpoint creation time
- Overall execution time

## Notes

- The checkpoint includes container state but ignores volumes (--ignore-volumes)
- GPU memory utilization is set to 0.90 (90%)
- Maximum model length is 4096 tokens
- The script requires privileged mode for GPU access
- seccomp profile is configured to disable io_uring
