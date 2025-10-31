#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///

"""Benchmark vLLM container restore with VMM lazy loading.

This script measures Time-to-First-Token (TTFT) when restoring a vLLM
container using CRIU's VMM (Virtual Memory Management) lazy restore feature.

The VMM feature allows:
- Fast startup: Only load first N MB of GPU memory eagerly
- Background loading: Remaining memory loaded in parallel
- Reduced TTFT: Container responds before all memory is loaded

Usage:
    ./benchmark-vmm-restore.py --container vllm-llm-demo --eager-mb 100
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def run_command(cmd: list[str], env: Optional[dict] = None, capture: bool = False) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command with error handling."""
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    try:
        if capture:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            return result
        else:
            subprocess.run(cmd, check=True, env=env)
            return None
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed with exit code {e.returncode}[/red]")
        if capture and e.stderr:
            console.print(f"[red]Error: {e.stderr}[/red]")
        raise


def wait_for_health(url: str = "http://localhost:8000/health", timeout: int = 120) -> float:
    """Wait for vLLM health endpoint to respond.

    Returns:
        Time in seconds until health check succeeded
    """
    start = time.time()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Waiting for vLLM health check...", total=None)

        while time.time() - start < timeout:
            try:
                response = httpx.get(url, timeout=2.0)
                if response.status_code == 200:
                    elapsed = time.time() - start
                    progress.update(task, description=f"[green]✓ Health check OK ({elapsed:.2f}s)")
                    return elapsed
            except (httpx.RequestError, httpx.TimeoutException):
                pass
            time.sleep(0.5)

    raise TimeoutError(f"Health check did not succeed within {timeout}s")


def measure_ttft(prompt: str, api_url: str = "http://localhost:8000/v1/completions") -> tuple[float, str]:
    """Measure Time-to-First-Token for a completion request.

    Returns:
        Tuple of (ttft_seconds, full_response_text)
    """
    payload = {
        "model": "Qwen/Qwen2-1.5B-Instruct",
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True,
    }

    start = time.time()
    first_token_time = None
    response_text = ""

    with httpx.stream("POST", api_url, json=payload, timeout=60.0) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line or line == "data: [DONE]":
                continue

            if line.startswith("data: "):
                data = json.loads(line[6:])

                if first_token_time is None:
                    first_token_time = time.time()

                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0].get("text", "")
                    response_text += text

    ttft = first_token_time - start if first_token_time else 0.0
    return ttft, response_text


def benchmark_vmm_restore(container_name: str, eager_mb: int, workers: int) -> dict:
    """Benchmark VMM-enabled container restore.

    Returns:
        Dictionary with timing metrics
    """
    results = {
        "container_name": container_name,
        "vmm_enabled": True,
        "eager_mb": eager_mb,
        "workers": workers,
        "timestamp": datetime.now().isoformat(),
    }

    # Setup environment for VMM
    env = os.environ.copy()
    env["CRIU_CUDA_VMM_LAZY"] = "1"
    env["CRIU_CUDA_VMM_EAGER_MB"] = str(eager_mb)
    env["CRIU_CUDA_VMM_WORKERS"] = str(workers)

    console.print("\n[bold cyan]═══ VMM Lazy Restore Benchmark ═══[/bold cyan]")
    console.print(f"Container: {container_name}")
    console.print(f"Eager size: {eager_mb} MB")
    console.print(f"Workers: {workers}")
    console.print()

    # T0: Start restore
    console.print("[yellow]⏱ T0: Starting container restore...[/yellow]")
    t0 = time.time()

    try:
        run_command(
            ["podman", "container", "restore", "--keep", container_name],
            env=env
        )
        t1 = time.time()
        results["restore_time"] = t1 - t0
        console.print(f"[green]✓ T1: Container restored ({results['restore_time']:.2f}s)[/green]")

        # T2: Wait for health
        console.print("[yellow]⏱ T2: Waiting for health check...[/yellow]")
        health_time = wait_for_health()
        t2 = time.time()
        results["health_time"] = health_time
        results["total_ready_time"] = t2 - t0
        console.print(f"[green]✓ T2: Health check OK ({health_time:.2f}s)[/green]")

        # T3: Measure TTFT
        console.print("[yellow]⏱ T3: Measuring TTFT...[/yellow]")
        prompt = "Write a short poem about GPUs:"
        ttft, response = measure_ttft(prompt)
        t3 = time.time()
        results["ttft"] = ttft
        results["total_ttft"] = t3 - t0
        console.print(f"[green]✓ T3: First token ({ttft:.3f}s)[/green]")
        console.print(f"[dim]Response: {response[:100]}...[/dim]")

        # Success
        results["success"] = True

    except Exception as e:
        console.print(f"[red]✗ Benchmark failed: {e}[/red]")
        results["success"] = False
        results["error"] = str(e)

    finally:
        # Stop container
        console.print("\n[yellow]Stopping container...[/yellow]")
        try:
            run_command(["podman", "stop", container_name])
        except:
            pass

    return results


def print_results(results: dict):
    """Print benchmark results in a nice table."""
    table = Table(title="VMM Restore Benchmark Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if results["success"]:
        table.add_row("Container Restore", f"{results['restore_time']:.2f}s")
        table.add_row("Health Check Ready", f"{results['health_time']:.2f}s")
        table.add_row("TTFT (Time to First Token)", f"{results['ttft']:.3f}s")
        table.add_row("Total Time (T0 → T3)", f"{results['total_ttft']:.2f}s")
        table.add_row("", "")
        table.add_row("VMM Configuration", "")
        table.add_row("  Eager Size", f"{results['eager_mb']} MB")
        table.add_row("  Workers", str(results['workers']))
    else:
        table.add_row("Status", "[red]FAILED[/red]")
        table.add_row("Error", results.get("error", "Unknown"))

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Benchmark VMM-enabled vLLM restore")
    parser.add_argument(
        "--container",
        default="vllm-llm-demo",
        help="Container name to restore (default: vllm-llm-demo)"
    )
    parser.add_argument(
        "--eager-mb",
        type=int,
        default=100,
        help="Amount of GPU memory to load eagerly in MB (default: 100)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads for lazy loading (default: 4)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_vmm_restore(args.container, args.eager_mb, args.workers)

    # Print results
    print_results(results)

    # Save to file
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to: {args.output}[/green]")

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
