#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx",
#   "rich",
# ]
# ///

"""
Shared utility functions for GPU benchmark scripts.
Provides health checks, inference timing, container management, and logging.
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def get_timestamp_ns() -> int:
    """
    Get current timestamp in nanoseconds using high-resolution performance counter.

    Returns:
        int: Current timestamp in nanoseconds
    """
    return time.perf_counter_ns()


def log_metric(label: str, value: Any, unit: str = "") -> None:
    """
    Log a metric with structured formatting and timestamp.

    Args:
        label: Metric label/name
        value: Metric value
        unit: Optional unit string (e.g., "ms", "seconds", "requests")
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    unit_str = f" {unit}" if unit else ""
    console.print(f"[dim]{timestamp}[/dim] [cyan]{label}:[/cyan] [bold]{value}{unit_str}[/bold]")


def load_env(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file (default: ".env")

    Returns:
        Dict[str, str]: Dictionary of environment variables
    """
    env_path = Path(env_file)
    env_vars = {}

    if not env_path.exists():
        console.print(f"[yellow]Warning: {env_file} not found[/yellow]")
        return env_vars

    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Parse KEY=VALUE format
            if "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                env_vars[key.strip()] = value

    console.print(f"[green]Loaded {len(env_vars)} environment variables from {env_file}[/green]")
    return env_vars


def wait_for_health(port: int, timeout: int = 300) -> float:
    """
    Poll vLLM /health endpoint until it returns 200 OK.

    Args:
        port: Port number where vLLM is running
        timeout: Maximum time to wait in seconds (default: 300)

    Returns:
        float: Time taken to become healthy in seconds

    Raises:
        TimeoutError: If health check doesn't succeed within timeout
        Exception: For other connection errors
    """
    health_url = f"http://localhost:{port}/health"
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Waiting for vLLM health check on port {port}...", total=None)

        with httpx.Client(timeout=10.0) as client:
            while True:
                elapsed = time.time() - start_time

                if elapsed > timeout:
                    raise TimeoutError(
                        f"Health check timed out after {timeout} seconds on port {port}"
                    )

                try:
                    response = client.get(health_url)
                    if response.status_code == 200:
                        elapsed_time = time.time() - start_time
                        console.print(
                            f"[green]✓ vLLM is healthy on port {port}[/green] "
                            f"[dim](took {elapsed_time:.2f}s)[/dim]"
                        )
                        return elapsed_time
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Connection refused or timeout - service not ready yet
                    pass
                except Exception as e:
                    console.print(f"[yellow]Health check error: {e}[/yellow]")

                time.sleep(2)


def send_inference(
    port: int,
    prompt: str = "Tell me a story",
    model: str = "meta-llama/Llama-3.2-1B-Instruct",
    max_tokens: int = 100,
) -> Dict[str, float]:
    """
    Send inference request to vLLM and measure timing metrics.

    Args:
        port: Port number where vLLM is running
        prompt: Input prompt for inference
        model: Model name to use
        max_tokens: Maximum tokens to generate

    Returns:
        Dict with timing metrics:
            - tfft: Time to first token in seconds
            - total_time: Total inference time in seconds
            - tokens_generated: Number of tokens generated

    Raises:
        Exception: If inference request fails
    """
    inference_url = f"http://localhost:{port}/v1/completions"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }

    console.print(f"[cyan]Sending inference request to port {port}...[/cyan]")
    console.print(f"[dim]Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}[/dim]")

    start_time = time.time()
    tfft = None
    tokens_generated = 0

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", inference_url, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.strip():
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        if data_str.strip() == "[DONE]":
                            break

                        # Record TFFT on first token
                        if tfft is None:
                            tfft = time.time() - start_time

                        tokens_generated += 1

        total_time = time.time() - start_time

        # Create results table
        table = Table(title="Inference Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("TFFT (Time to First Token)", f"{tfft:.3f}s" if tfft else "N/A")
        table.add_row("Total Time", f"{total_time:.3f}s")
        table.add_row("Tokens Generated", str(tokens_generated))
        if tokens_generated > 0 and total_time > 0:
            table.add_row("Tokens/Second", f"{tokens_generated / total_time:.2f}")

        console.print(table)

        return {
            "tfft": tfft or 0.0,
            "total_time": total_time,
            "tokens_generated": tokens_generated,
        }

    except httpx.HTTPStatusError as e:
        console.print(f"[red]HTTP error during inference: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Inference error: {e}[/red]")
        raise


def cleanup_containers(container_name: str) -> None:
    """
    Forcefully remove a podman container.

    Args:
        container_name: Name of the container to remove
    """
    console.print(f"[yellow]Cleaning up container: {container_name}[/yellow]")

    try:
        # Stop the container first (ignore errors if already stopped)
        subprocess.run(
            ["podman", "stop", container_name],
            capture_output=True,
            timeout=30,
        )

        # Force remove the container
        result = subprocess.run(
            ["podman", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            console.print(f"[green]✓ Container {container_name} removed successfully[/green]")
        else:
            # Container might not exist, which is fine
            if "no such container" in result.stderr.lower():
                console.print(f"[dim]Container {container_name} does not exist (already cleaned)[/dim]")
            else:
                console.print(f"[yellow]Warning: {result.stderr}[/yellow]")

    except subprocess.TimeoutExpired:
        console.print(f"[red]Timeout while removing container {container_name}[/red]")
    except Exception as e:
        console.print(f"[red]Error cleaning up container: {e}[/red]")


def run_command(
    cmd: list[str] | str,
    description: str,
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a shell command with timing and error handling.

    Args:
        cmd: Command to run (list of args or string)
        description: Human-readable description of the command
        timeout: Optional timeout in seconds
        check: Whether to raise exception on non-zero exit code

    Returns:
        subprocess.CompletedProcess: Result of the command

    Raises:
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If command times out
    """
    console.print(Panel(
        f"[bold]{description}[/bold]\n[dim]{' '.join(cmd) if isinstance(cmd, list) else cmd}[/dim]",
        border_style="blue"
    ))

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
            shell=isinstance(cmd, str),
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            console.print(f"[green]✓ {description} completed successfully[/green] [dim]({elapsed:.2f}s)[/dim]")
        else:
            console.print(f"[red]✗ {description} failed with exit code {result.returncode}[/red]")

        # Show output if present
        if result.stdout.strip():
            console.print("[dim]stdout:[/dim]")
            console.print(result.stdout)
        if result.stderr.strip():
            console.print("[dim]stderr:[/dim]")
            console.print(result.stderr)

        return result

    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start_time
        console.print(f"[red]✗ {description} timed out after {elapsed:.2f}s[/red]")
        raise
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        console.print(f"[red]✗ {description} failed after {elapsed:.2f}s[/red]")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        console.print(f"[red]✗ {description} error after {elapsed:.2f}s: {e}[/red]")
        raise


if __name__ == "__main__":
    """
    Test the utility functions when run directly.
    """
    console.print(Panel.fit(
        "[bold cyan]GPU Load Utils Test Suite[/bold cyan]",
        border_style="cyan"
    ))

    # Test logging
    log_metric("Test Metric", 42, "units")
    log_metric("Another Test", 3.14159)

    # Test timestamp
    ts = get_timestamp_ns()
    console.print(f"\n[cyan]Timestamp (ns):[/cyan] {ts}")

    # Test env loading
    console.print("\n[bold]Testing environment loading:[/bold]")
    env_vars = load_env()
    if env_vars:
        for key, value in list(env_vars.items())[:3]:  # Show first 3
            console.print(f"  {key} = {value}")

    # Test command execution
    console.print("\n[bold]Testing command execution:[/bold]")
    try:
        result = run_command(["echo", "Hello from utils.py"], "Echo test", timeout=5)
        console.print(f"[green]Command succeeded with output: {result.stdout.strip()}[/green]")
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")

    console.print("\n[green]✓ Utils module test complete![/green]")
