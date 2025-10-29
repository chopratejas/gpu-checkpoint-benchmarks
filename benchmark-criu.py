#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
#     "rich",
#     "python-dotenv",
# ]
# ///

"""
CRIU Restore TFFT Benchmark

Measures time to first token when restoring a container from a CRIU checkpoint.
Tracks multiple timing points to understand the restore and inference pipeline.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_environment() -> Dict[str, str]:
    """Load and validate environment variables from .env file."""
    load_dotenv()

    required_vars = ["CONT_NAME", "RESULTS_DIR", "API_PORT"]
    env_vars = {}

    for var in required_vars:
        value = os.getenv(var)
        if not value:
            console.print(f"[red]Error: {var} not set in .env file[/red]")
            sys.exit(1)
        env_vars[var] = value

    # Optional variables with defaults
    env_vars["MODEL_NAME"] = os.getenv("MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
    env_vars["HEALTH_TIMEOUT"] = int(os.getenv("HEALTH_TIMEOUT", "300"))
    env_vars["HEALTH_INTERVAL"] = float(os.getenv("HEALTH_INTERVAL", "0.5"))

    return env_vars


def verify_checkpoint(cont_name: str) -> None:
    """Verify that the checkpoint exists for container."""
    # Check if container exists
    result = subprocess.run(
        ["podman", "inspect", "--format", "{{.Id}}", cont_name],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        console.print(f"[red]Error: Container {cont_name} not found[/red]")
        console.print(f"[yellow]Did you run create-checkpoint.py first?[/yellow]")
        sys.exit(1)

    container_id = result.stdout.strip()
    ckpt_dir = Path(f"/var/lib/containers/storage/overlay-containers/{container_id}/userdata/checkpoint")

    if not ckpt_dir.exists():
        console.print(f"[red]Error: No checkpoint found for container {cont_name}[/red]")
        console.print(f"[yellow]Expected location: {ckpt_dir}[/yellow]")
        console.print(f"[yellow]Run create-checkpoint.py first to create checkpoint[/yellow]")
        sys.exit(1)

    console.print(f"[green]Checkpoint verified for container: {cont_name}[/green]")


def cleanup_containers() -> None:
    """Stop and remove any existing containers."""
    console.print("[yellow]Cleaning up existing containers...[/yellow]")

    # Get list of all containers
    result = subprocess.run(
        ["podman", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and result.stdout.strip():
        containers = result.stdout.strip().split('\n')
        for container in containers:
            if container:
                subprocess.run(["podman", "rm", "-f", container],
                             capture_output=True)
                console.print(f"[dim]Removed container: {container}[/dim]")


def restore_container(cont_name: str) -> tuple[float, float]:
    """
    Restore container from LOCAL checkpoint (no tar import).

    Args:
        cont_name: Container name to restore

    Returns:
        Tuple of (T0, T1) timestamps
    """
    console.print(f"[cyan]Restoring container from LOCAL checkpoint: {cont_name}[/cyan]")

    # T0: Start timer before restore
    t0 = time.perf_counter()

    result = subprocess.run(
        ["podman", "container", "restore",
         "--keep",  # Keep checkpoint for multiple restores
         cont_name],
        capture_output=True,
        text=True
    )

    # T1: Container restored
    t1 = time.perf_counter()

    if result.returncode != 0:
        console.print(f"[red]Error restoring container:[/red]")
        console.print(result.stderr)
        sys.exit(1)

    restore_time = t1 - t0
    console.print(f"[green]Container restored in {restore_time:.3f}s[/green]")

    return t0, t1


def wait_for_health(api_port: str, timeout: int, interval: float) -> float:
    """
    Wait for the /health endpoint to return successfully.

    Returns:
        T2 timestamp when API is ready
    """
    health_url = f"http://localhost:{api_port}/health"
    console.print(f"[cyan]Waiting for API health check at {health_url}...[/cyan]")

    start_time = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Checking health endpoint...", total=None)

        while time.perf_counter() - start_time < timeout:
            try:
                response = httpx.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    t2 = time.perf_counter()
                    elapsed = t2 - start_time
                    console.print(f"[green]API ready in {elapsed:.3f}s[/green]")
                    return t2
            except (httpx.ConnectError, httpx.TimeoutException):
                pass

            time.sleep(interval)

    console.print(f"[red]Error: Health check timeout after {timeout}s[/red]")
    sys.exit(1)


def measure_inference_tfft(api_port: str, model_name: str) -> tuple[float, float, float]:
    """
    Send inference request and measure time to first token.

    Returns:
        Tuple of (T2_inference_start, T3_first_token, T4_completion)
    """
    inference_url = f"http://localhost:{api_port}/v1/completions"
    console.print("[cyan]Sending inference request...[/cyan]")

    payload = {
        "model": model_name,
        "prompt": "What is the capital of France?",
        "max_tokens": 50,
        "stream": True
    }

    t2_inference = time.perf_counter()
    t3 = None
    t4 = None

    try:
        with httpx.stream(
            "POST",
            inference_url,
            json=payload,
            timeout=30.0
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_lines():
                if chunk.strip():
                    if t3 is None:
                        # First token received
                        t3 = time.perf_counter()
                        tfft = t3 - t2_inference
                        console.print(f"[green]First token received in {tfft:.3f}s[/green]")

            # Full response received
            t4 = time.perf_counter()
            full_time = t4 - t2_inference
            console.print(f"[green]Full response received in {full_time:.3f}s[/green]")

    except Exception as e:
        console.print(f"[red]Error during inference: {e}[/red]")
        sys.exit(1)

    if t3 is None:
        console.print("[red]Error: No tokens received from inference[/red]")
        sys.exit(1)

    return t2_inference, t3, t4


def save_results(env_vars: Dict[str, str], metrics: Dict[str, Any]) -> None:
    """Save benchmark results to JSON file."""
    results_dir = Path(env_vars["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"criu_benchmark_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]Results saved to: {results_file}[/green]")


def cleanup_final() -> None:
    """Final cleanup of containers."""
    console.print("[yellow]Final cleanup...[/yellow]")
    cleanup_containers()


def main():
    """Main benchmark execution."""
    console.rule("[bold blue]CRIU Restore TFFT Benchmark")

    # 1. Load environment
    console.print("\n[bold]Step 1: Loading environment[/bold]")
    env_vars = load_environment()

    # 2. Verify checkpoint
    console.print("\n[bold]Step 2: Verifying checkpoint[/bold]")
    verify_checkpoint(env_vars["CONT_NAME"])

    # 3. Note: No cleanup needed - checkpoint restore handles this
    console.print("\n[bold]Step 3: Preparing for restore[/bold]")
    console.print(f"[cyan]Using LOCAL checkpoint for container: {env_vars['CONT_NAME']}[/cyan]")

    try:
        # 4. Restore container and measure T0, T1
        console.print("\n[bold]Step 4: Restoring container from LOCAL checkpoint[/bold]")
        t0, t1 = restore_container(env_vars["CONT_NAME"])

        # 5. Wait for health check and measure T2
        console.print("\n[bold]Step 5: Waiting for API health check[/bold]")
        t2 = wait_for_health(
            env_vars["API_PORT"],
            env_vars["HEALTH_TIMEOUT"],
            env_vars["HEALTH_INTERVAL"]
        )

        # 6. Measure inference TFFT (T3, T4)
        console.print("\n[bold]Step 6: Measuring inference TFFT[/bold]")
        t2_inference, t3, t4 = measure_inference_tfft(
            env_vars["API_PORT"],
            env_vars["MODEL_NAME"]
        )

        # 7. Calculate metrics
        console.print("\n[bold]Step 7: Calculating metrics[/bold]")
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "type": "criu",
            "model": env_vars["MODEL_NAME"],
            "restore_time": round(t1 - t0, 3),
            "reinitialization_time": round(t2 - t1, 3),
            "inference_tfft": round(t3 - t2_inference, 3),
            "total_criu_tfft": round(t3 - t0, 3),
            "full_response_time": round(t4 - t0, 3)
        }

        # 8. Display results
        console.print("\n[bold green]Benchmark Results:[/bold green]")
        console.print(f"  Restore time (T1-T0):          {metrics['restore_time']:.3f}s")
        console.print(f"  Reinitialization (T2-T1):      {metrics['reinitialization_time']:.3f}s")
        console.print(f"  Inference TFFT (T3-T2):        {metrics['inference_tfft']:.3f}s")
        console.print(f"  [bold]Total CRIU TFFT (T3-T0):       {metrics['total_criu_tfft']:.3f}s[/bold]")
        console.print(f"  Full response time (T4-T0):    {metrics['full_response_time']:.3f}s")

        # 9. Save results
        console.print("\n[bold]Step 8: Saving results[/bold]")
        save_results(env_vars, metrics)

    finally:
        # 10. Cleanup
        console.print("\n[bold]Step 9: Cleanup[/bold]")
        cleanup_final()

    console.rule("[bold green]Benchmark Complete")


if __name__ == "__main__":
    main()
