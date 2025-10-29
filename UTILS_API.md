# Utils.py API Reference

A Python utility module with PEP 723 inline dependencies for vLLM benchmarking.

## Installation

The module uses `uv` for dependency management. Dependencies are automatically installed when running with the shebang:

```bash
./utils.py  # Runs tests and examples
```

## Dependencies (Auto-managed via PEP 723)

- `httpx` - Modern HTTP client for API requests
- `rich` - Beautiful terminal output and formatting

## Functions

### 1. `get_timestamp_ns() -> int`

Get current high-resolution timestamp in nanoseconds.

**Returns:** Current timestamp in nanoseconds using `time.perf_counter_ns()`

**Example:**
```python
from utils import get_timestamp_ns
start = get_timestamp_ns()
# ... do work ...
elapsed_ns = get_timestamp_ns() - start
print(f"Elapsed: {elapsed_ns / 1e9:.3f}s")
```

---

### 2. `log_metric(label: str, value: Any, unit: str = "") -> None`

Log a metric with structured formatting and timestamp.

**Parameters:**
- `label`: Metric label/name
- `value`: Metric value (any type)
- `unit`: Optional unit string (e.g., "ms", "seconds", "%")

**Example:**
```python
from utils import log_metric
log_metric("CPU Usage", 42.5, "%")
log_metric("Latency", 123.4, "ms")
log_metric("Status", "OK")
```

**Output:**
```
2025-10-28 22:43:39.751 CPU Usage: 42.5 %
2025-10-28 22:43:39.752 Latency: 123.4 ms
2025-10-28 22:43:39.753 Status: OK
```

---

### 3. `load_env(env_file: str = ".env") -> Dict[str, str]`

Load environment variables from .env file.

**Parameters:**
- `env_file`: Path to .env file (default: ".env")

**Returns:** Dictionary of environment variables

**Example:**
```python
from utils import load_env
env = load_env()
model_id = env.get("MODEL_ID", "default-model")
port = int(env.get("API_PORT", "8000"))
```

**Supported .env format:**
```bash
# Comments are ignored
KEY=value
QUOTED="value with spaces"
SINGLE='quoted value'
```

---

### 4. `wait_for_health(port: int, timeout: int = 300) -> float`

Poll vLLM /health endpoint until it returns 200 OK.

**Parameters:**
- `port`: Port number where vLLM is running
- `timeout`: Maximum time to wait in seconds (default: 300)

**Returns:** Time taken to become healthy in seconds

**Raises:**
- `TimeoutError`: If health check doesn't succeed within timeout
- `Exception`: For other connection errors

**Example:**
```python
from utils import wait_for_health

# Wait for vLLM to be ready
elapsed = wait_for_health(port=8000, timeout=300)
print(f"vLLM ready in {elapsed:.2f}s")
```

---

### 5. `send_inference(port: int, prompt: str = "Tell me a story", model: str = "meta-llama/Llama-3.2-1B-Instruct", max_tokens: int = 100) -> Dict[str, float]`

Send inference request to vLLM and measure timing metrics.

**Parameters:**
- `port`: Port number where vLLM is running
- `prompt`: Input prompt for inference
- `model`: Model name to use
- `max_tokens`: Maximum tokens to generate

**Returns:** Dictionary with timing metrics:
- `tfft`: Time to first token in seconds
- `total_time`: Total inference time in seconds
- `tokens_generated`: Number of tokens generated

**Example:**
```python
from utils import send_inference

results = send_inference(
    port=8000,
    prompt="Explain quantum computing",
    max_tokens=200
)

print(f"TFFT: {results['tfft']:.3f}s")
print(f"Total: {results['total_time']:.3f}s")
print(f"Tokens: {results['tokens_generated']}")
```

---

### 6. `cleanup_containers(container_name: str) -> None`

Forcefully remove a podman container.

**Parameters:**
- `container_name`: Name of the container to remove

**Example:**
```python
from utils import cleanup_containers

# Clean up after benchmark
cleanup_containers("vllm-checkpoint")
```

**Note:** Handles non-existent containers gracefully (no error).

---

### 7. `run_command(cmd: list[str] | str, description: str, timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess`

Run a shell command with timing and error handling.

**Parameters:**
- `cmd`: Command to run (list of args or string)
- `description`: Human-readable description of the command
- `timeout`: Optional timeout in seconds
- `check`: Whether to raise exception on non-zero exit code

**Returns:** `subprocess.CompletedProcess` result

**Raises:**
- `subprocess.CalledProcessError`: If command fails and check=True
- `subprocess.TimeoutExpired`: If command times out

**Example:**
```python
from utils import run_command

# Run with list (safer, no shell injection)
result = run_command(
    ["podman", "ps", "-a"],
    "List all containers",
    timeout=10
)

# Run with string (uses shell)
result = run_command(
    "ls -la | grep py",
    "Find Python files",
    timeout=5
)

print(result.stdout)
```

---

## Usage in Benchmark Scripts

### Basic Import
```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx",
#   "rich",
# ]
# ///

import sys
sys.path.insert(0, '/root/gpu-load')

from utils import (
    wait_for_health,
    send_inference,
    cleanup_containers,
    log_metric,
    get_timestamp_ns,
    load_env,
    run_command,
)
```

### Complete Benchmark Example
```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["httpx", "rich"]
# ///

import sys
sys.path.insert(0, '/root/gpu-load')
from utils import *

# Load configuration
env = load_env()
port = int(env.get("API_PORT", "8000"))

try:
    # Start container
    run_command(
        ["podman", "run", "-d", "--name", "vllm-test", "..."],
        "Start vLLM container"
    )
    
    # Wait for health
    health_time = wait_for_health(port=port, timeout=300)
    log_metric("Health Check Time", health_time, "seconds")
    
    # Run inference
    start_ns = get_timestamp_ns()
    results = send_inference(port=port, prompt="Hello world")
    elapsed_ns = get_timestamp_ns() - start_ns
    
    log_metric("TFFT", results['tfft'], "s")
    log_metric("Total Time", results['total_time'], "s")
    log_metric("Throughput", results['tokens_generated'] / results['total_time'], "tok/s")
    
finally:
    # Cleanup
    cleanup_containers("vllm-test")
```

---

## Testing

Run the test suite:

```bash
./test_utils.py
```

Or run the built-in tests:

```bash
./utils.py
```

---

## Features

✅ **PEP 723 Inline Dependencies** - No separate requirements.txt needed  
✅ **Rich Terminal Output** - Colored, formatted console output  
✅ **Proper Error Handling** - Comprehensive exception handling  
✅ **Timeout Support** - All network operations have timeouts  
✅ **Streaming Support** - TFFT measurement via streaming inference  
✅ **High-Resolution Timing** - Nanosecond precision with `perf_counter_ns()`  
✅ **Structured Logging** - Timestamped, formatted metrics  
✅ **Container Management** - Safe cleanup with graceful error handling  

---

## Requirements

- Python 3.10+
- `uv` package manager (auto-installs dependencies)
- `podman` (for container operations)

## Installation of uv

If uv is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

## File Location

`/root/gpu-load/utils.py`
