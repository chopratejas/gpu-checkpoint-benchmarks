#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///

"""Create a vLLM container checkpoint after warmup inference requests.

This script:
1. Loads configuration from .env
2. Removes any existing containers with the same name
3. Starts a vLLM container with podman
4. Waits for health check
5. Runs warmup inference requests
6. Creates a checkpoint for fast container restoration
7. Verifies checkpoint and cleans up
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

import utils

console = Console()


def run_command(cmd: list[str], check: bool = True, capture: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command with error handling.

    Args:
        cmd: Command and arguments as list
        check: Raise exception on non-zero exit
        capture: Capture stdout/stderr

    Returns:
        CompletedProcess if capture=True, None otherwise
    """
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        if capture:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result
        else:
            subprocess.run(cmd, check=check)
            return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed with exit code {e.returncode}[/red]")
        if capture and e.stderr:
            console.print(f"[red]Error: {e.stderr}[/red]")
        raise


def remove_existing_container(container_name: str) -> None:
    """Remove existing container if it exists.

    Args:
        container_name: Name of container to remove
    """
    console.print(f"[yellow]Checking for existing container: {container_name}[/yellow]")

    # Check if container exists
    result = subprocess.run(
        ["podman", "ps", "-a", "--filter", f"name=^{container_name}$", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if result.stdout.strip():
        console.print(f"[yellow]Removing existing container: {container_name}[/yellow]")
        # Force remove (stop if running, then remove)
        run_command(["podman", "rm", "-f", container_name], check=False)
        time.sleep(1)
    else:
        console.print(f"[green]No existing container found[/green]")


def start_vllm_container(env_vars: dict[str, str]) -> None:
    """Start vLLM container with exact configuration.

    Args:
        env_vars: Environment variables from .env
    """
    cont_name = env_vars.get("CONT_NAME", "vllm-checkpoint")
    model_id = env_vars.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    api_port = env_vars.get("API_PORT", "8000")

    console.print(f"[cyan]Starting vLLM container: {cont_name}[/cyan]")
    console.print(f"[cyan]Model: {model_id}[/cyan]")
    console.print(f"[cyan]Port: {api_port}[/cyan]")

    cmd = [
        "podman", "run", "-d",
        "--name", cont_name,
        "--device", "/dev/null:/dev/null:rwm",
        "--privileged",
        "--security-opt=label=disable",
        "--security-opt", "seccomp=/etc/containers/seccomp.d/no-io-uring.json",
        "--device", "/dev/nvidia0",
        "--device", "/dev/nvidiactl",
        "--device", "/dev/nvidia-uvm",
        "--shm-size", "8g",
        "-e", f"LD_LIBRARY_PATH=/opt/nvidia-libs:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "-e", "ASYNCIO_DEFAULT_BACKEND=select",
        "-e", "PYTHON_ASYNCIO_NO_IO_URING=1",
        "-v", "/opt/nvidia-libs:/opt/nvidia-libs:ro",
        "-v", "/models:/root/.cache/huggingface",
        "-p", f"{api_port}:{api_port}",
        "docker.io/vllm/vllm-openai:latest",
        "--model", model_id,
        "--host", "0.0.0.0",
        "--port", api_port,
        "--gpu-memory-utilization", "0.30",  # Further reduced to 0.30 for faster restore
        "--max-model-len", "1024",  # Reduced to 1024 for minimal KV cache
        "--trust-remote-code",
        "--load-format", "safetensors",  # Use mmap for file-backed memory
        "--enforce-eager"  # Disable CUDA graphs for CRIU compatibility
    ]

    result = run_command(cmd, capture=True)
    container_id = result.stdout.strip()
    console.print(f"[green]Container started: {container_id[:12]}[/green]")


def run_warmup_requests(num_requests: int = 5, port: int = 8000, model: str = "meta-llama/Llama-3.2-1B-Instruct") -> None:
    """Run warmup inference requests.

    Args:
        num_requests: Number of warmup requests to send
        port: Port to send to
        model: Model name to use for inference
    """
    console.print(f"[cyan]Running {num_requests} warmup inference requests...[/cyan]")

    prompts = [
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of containerization?",
        "Describe the concept of neural networks."
    ]

    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Warmup requests...", total=num_requests)

        for i in range(num_requests):
            try:
                prompt = prompts[i % len(prompts)]
                result = utils.send_inference(port=port, prompt=prompt, model=model, max_tokens=50)

                console.print(f"[green]Request {i+1}/{num_requests} completed[/green]")
                console.print(f"[dim]  TFFT: {result['tfft']:.3f}s, Total: {result['total_time']:.3f}s, Tokens: {result['tokens_generated']}[/dim]")
                progress.advance(task)
            except Exception as e:
                console.print(f"[red]Request {i+1} failed: {e}[/red]")
                raise


def create_checkpoint(env_vars: dict[str, str]) -> tuple[str, float]:
    """Create container checkpoint using LOCAL storage (no tar export).

    Args:
        env_vars: Environment variables from .env

    Returns:
        Tuple of (container_name, creation_time_seconds)
    """
    cont_name = env_vars.get("CONT_NAME", "vllm-checkpoint")

    console.print(f"[cyan]Creating LOCAL checkpoint for: {cont_name}[/cyan]")
    console.print(f"[cyan]Using Podman internal storage (no tar export)[/cyan]")

    start_time = time.time()

    cmd = [
        "podman", "container", "checkpoint",
        "--print-stats",  # Show detailed timing breakdown
        cont_name
    ]
    # NOTE: Not using --keep so container STOPS and GPU memory is freed
    # This ensures we measure TRUE cold start on restore

    run_command(cmd)

    creation_time = time.time() - start_time
    console.print(f"[green]Checkpoint created in {creation_time:.2f}s[/green]")

    return cont_name, creation_time


def verify_checkpoint(cont_name: str) -> None:
    """Verify checkpoint exists and get its location.

    Args:
        cont_name: Container name
    """
    # Get container ID to find checkpoint location
    result = subprocess.run(
        ["podman", "inspect", "--format", "{{.Id}}", cont_name],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        console.print(f"[yellow]Warning: Could not inspect container {cont_name}[/yellow]")
        return

    container_id = result.stdout.strip()

    # Checkpoint stored in Podman's internal storage
    ckpt_dir = Path(f"/var/lib/containers/storage/overlay-containers/{container_id}/userdata/checkpoint")

    if ckpt_dir.exists():
        # Calculate total size of checkpoint directory
        total_size = sum(f.stat().st_size for f in ckpt_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        size_gb = total_size / (1024 * 1024 * 1024)

        if size_gb >= 1:
            size_str = f"{size_gb:.2f} GB"
        else:
            size_str = f"{size_mb:.2f} MB"

        console.print(f"[green]Checkpoint verified: {ckpt_dir}[/green]")
        console.print(f"[green]Checkpoint size: {size_str}[/green]")
    else:
        console.print(f"[yellow]Warning: Checkpoint directory not found at {ckpt_dir}[/yellow]")


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    console.print("[bold blue]vLLM Checkpoint Creation Script[/bold blue]")
    console.print()

    overall_start = time.time()

    try:
        # Step 1: Load environment configuration
        console.print("[bold]Step 1: Loading environment configuration[/bold]")
        env_vars = utils.load_env()
        if not env_vars:
            console.print("[red]Failed to load .env file[/red]")
            return 1

        cont_name = env_vars.get("CONT_NAME", "vllm-checkpoint")
        api_port = int(env_vars.get("API_PORT", "8000"))

        console.print(f"[green]Configuration loaded:[/green]")
        console.print(f"  Container name: {cont_name}")
        console.print(f"  API port: {api_port}")
        console.print(f"  Checkpoint: LOCAL (Podman internal storage)")
        console.print()

        # Step 2: Remove existing container
        console.print("[bold]Step 2: Removing any existing containers[/bold]")
        remove_existing_container(cont_name)
        console.print()

        # Step 3: Start vLLM container
        console.print("[bold]Step 3: Starting vLLM container[/bold]")
        start_vllm_container(env_vars)
        console.print()

        # Step 4: Wait for health check
        console.print("[bold]Step 4: Waiting for health check[/bold]")
        try:
            health_time = utils.wait_for_health(port=api_port, timeout=300)
            console.print(f"[green]Health check passed in {health_time:.2f}s[/green]")
        except TimeoutError as e:
            console.print(f"[red]Health check failed: {e}[/red]")
            return 1
        console.print()

        # Step 5: Run warmup requests
        console.print("[bold]Step 5: Running warmup inference requests[/bold]")
        model_id = env_vars.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
        run_warmup_requests(num_requests=5, port=api_port, model=model_id)
        console.print()

        # Step 6: Create checkpoint
        console.print("[bold]Step 6: Creating LOCAL checkpoint[/bold]")
        cont_name_verified, creation_time = create_checkpoint(env_vars)
        console.print()

        # Step 7: Verify checkpoint
        console.print("[bold]Step 7: Verifying checkpoint[/bold]")
        verify_checkpoint(cont_name_verified)
        console.print()

        # Step 8: Note about container state
        console.print("[bold]Step 8: Container state[/bold]")
        console.print(f"[green]Container {cont_name} is checkpointed and STOPPED[/green]")
        console.print(f"[green]GPU memory has been FREED (true cold start)[/green]")
        console.print(f"[yellow]Note: Checkpoint preserved in Podman storage[/yellow]")
        console.print(f"[yellow]      Container can be restored multiple times[/yellow]")
        console.print()

        # Summary
        total_time = time.time() - overall_start
        console.print("[bold green]Checkpoint creation completed successfully![/bold green]")
        console.print(f"[green]Total time: {total_time:.2f}s[/green]")
        console.print(f"[green]Container: {cont_name}[/green]")
        console.print(f"[cyan]To restore: podman container restore --keep {cont_name}[/cyan]")
        console.print(f"[dim]Expected restore time: ~10-12s with TRUE GPU cold start[/dim]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
