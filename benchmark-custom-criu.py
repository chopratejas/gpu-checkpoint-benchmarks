#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///

"""Benchmark custom-built CRIU against system CRIU for vLLM restore."""

import os
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

import utils

console = Console()


def get_container_id(container_name: str) -> str | None:
    """Get full container ID from name."""
    result = subprocess.run(
        ["podman", "inspect", "--format", "{{.Id}}", container_name],
        capture_output=True,
        text=True,
        check=False
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return None


def cleanup_container(container_name: str) -> None:
    """Force remove container if it exists."""
    subprocess.run(
        ["podman", "rm", "-f", container_name],
        capture_output=True,
        check=False
    )
    time.sleep(1)


def restore_with_criu(criu_binary: str, container_name: str, keep: bool = True) -> tuple[float, bool]:
    """
    Restore container using specified CRIU binary.

    Args:
        criu_binary: Path to CRIU binary (or "system" for default podman)
        container_name: Container to restore
        keep: Whether to keep container running after restore

    Returns:
        Tuple of (restore_time_seconds, success)
    """
    console.print(f"[cyan]Restoring with CRIU: {criu_binary}[/cyan]")

    start_time = time.time()

    if criu_binary == "system":
        # Use default podman restore (system CRIU)
        cmd = ["podman", "container", "restore"]
        if keep:
            cmd.append("--keep")
        cmd.append(container_name)
    else:
        # Use custom CRIU binary via --runtime-opt
        # Tell podman to use our custom CRIU
        cmd = [
            "podman", "container", "restore",
            f"--runtime-opt=runtime_criu_path={criu_binary}"
        ]
        if keep:
            cmd.append("--keep")
        cmd.append(container_name)

    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    restore_time = time.time() - start_time

    if result.returncode != 0:
        console.print(f"[red]Restore failed![/red]")
        console.print(f"[red]Error: {result.stderr}[/red]")
        return restore_time, False

    console.print(f"[green]Restore completed in {restore_time:.2f}s[/green]")
    return restore_time, True


def verify_container_healthy(port: int = 8000, timeout: int = 60) -> tuple[float, bool]:
    """Check if container is healthy and accepting requests."""
    console.print(f"[cyan]Checking container health on port {port}...[/cyan]")

    try:
        health_time = utils.wait_for_health(port=port, timeout=timeout)
        console.print(f"[green]Container healthy in {health_time:.2f}s[/green]")
        return health_time, True
    except TimeoutError as e:
        console.print(f"[red]Health check failed: {e}[/red]")
        return 0.0, False


def run_benchmark_iteration(
    criu_binary: str,
    container_name: str,
    api_port: int,
    model: str,
    iteration: int
) -> dict:
    """Run single benchmark iteration."""
    console.print(f"\n[bold cyan]Iteration {iteration}[/bold cyan]")

    # 1. Clean up any existing container
    cleanup_container(container_name)

    # 2. Restore container
    restore_time, restore_success = restore_with_criu(criu_binary, container_name, keep=True)

    if not restore_success:
        return {
            "iteration": iteration,
            "restore_time": restore_time,
            "health_time": 0.0,
            "inference_time": 0.0,
            "total_time": restore_time,
            "success": False
        }

    # 3. Wait for health
    health_time, health_success = verify_container_healthy(port=api_port, timeout=120)

    if not health_success:
        cleanup_container(container_name)
        return {
            "iteration": iteration,
            "restore_time": restore_time,
            "health_time": health_time,
            "inference_time": 0.0,
            "total_time": restore_time + health_time,
            "success": False
        }

    # 4. Run test inference
    console.print(f"[cyan]Running test inference...[/cyan]")
    try:
        result = utils.send_inference(
            port=api_port,
            prompt="What is machine learning?",
            model=model,
            max_tokens=20
        )
        inference_time = result['total_time']
        console.print(f"[green]Inference completed in {inference_time:.3f}s[/green]")
        success = True
    except Exception as e:
        console.print(f"[red]Inference failed: {e}[/red]")
        inference_time = 0.0
        success = False

    # 5. Clean up
    cleanup_container(container_name)

    total_time = restore_time + health_time

    return {
        "iteration": iteration,
        "restore_time": restore_time,
        "health_time": health_time,
        "inference_time": inference_time,
        "total_time": total_time,
        "success": success
    }


def main() -> int:
    """Main benchmark function."""
    console.print("[bold blue]Custom CRIU Benchmark Script[/bold blue]")
    console.print()

    # Configuration
    env_vars = utils.load_env()
    if not env_vars:
        console.print("[red]Failed to load .env file[/red]")
        return 1

    container_name = env_vars.get("CONT_NAME", "vllm-checkpoint")
    api_port = int(env_vars.get("API_PORT", "8000"))
    model_id = env_vars.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    iterations = 3

    # Check if container has checkpoint
    container_id = get_container_id(container_name)
    if not container_id:
        console.print(f"[red]Container {container_name} not found![/red]")
        console.print("[yellow]Run ./create-checkpoint.py first[/yellow]")
        return 1

    console.print(f"[green]Container ID: {container_id[:12]}[/green]")

    # Check if checkpoint exists
    ckpt_dir = Path(f"/var/lib/containers/storage/overlay-containers/{container_id}/userdata/checkpoint")
    if not ckpt_dir.exists():
        console.print(f"[red]No checkpoint found at {ckpt_dir}[/red]")
        console.print("[yellow]Run ./create-checkpoint.py first[/yellow]")
        return 1

    console.print(f"[green]Checkpoint found: {ckpt_dir}[/green]")
    console.print()

    # Benchmarks to run
    benchmarks = [
        {
            "name": "System CRIU (baseline)",
            "criu_binary": "system",
        },
        {
            "name": "Custom CRIU (/root/criu/criu/criu)",
            "criu_binary": "/root/criu/criu/criu",
        },
    ]

    # Check if custom CRIU exists
    custom_criu = Path("/root/criu/criu/criu")
    if not custom_criu.exists():
        console.print(f"[red]Custom CRIU not found at {custom_criu}[/red]")
        console.print("[yellow]Build CRIU first: cd /root/criu && make[/yellow]")
        return 1

    results = {}

    # Run benchmarks
    for benchmark in benchmarks:
        bench_name = benchmark["name"]
        criu_binary = benchmark["criu_binary"]

        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print(f"[bold green]Benchmark: {bench_name}[/bold green]")
        console.print(f"[bold green]{'='*60}[/bold green]")

        bench_results = []

        for i in range(1, iterations + 1):
            result = run_benchmark_iteration(
                criu_binary=criu_binary,
                container_name=container_name,
                api_port=api_port,
                model=model_id,
                iteration=i
            )
            bench_results.append(result)
            time.sleep(2)  # Brief pause between iterations

        results[bench_name] = bench_results

    # Display results
    console.print(f"\n[bold green]{'='*60}[/bold green]")
    console.print(f"[bold green]Results Summary[/bold green]")
    console.print(f"[bold green]{'='*60}[/bold green]\n")

    table = Table(title="Restore Time Comparison")
    table.add_column("CRIU", style="cyan")
    table.add_column("Avg Restore (s)", justify="right", style="green")
    table.add_column("Min (s)", justify="right")
    table.add_column("Max (s)", justify="right")
    table.add_column("Speedup", justify="right", style="yellow")

    baseline_avg = None

    for bench_name, bench_results in results.items():
        successful = [r for r in bench_results if r["success"]]
        if not successful:
            table.add_row(bench_name, "FAILED", "-", "-", "-")
            continue

        restore_times = [r["restore_time"] for r in successful]
        avg = sum(restore_times) / len(restore_times)
        min_time = min(restore_times)
        max_time = max(restore_times)

        if baseline_avg is None:
            baseline_avg = avg
            speedup = "1.0x (baseline)"
        else:
            speedup_factor = baseline_avg / avg
            speedup = f"{speedup_factor:.2f}x"
            if speedup_factor > 1.0:
                speedup = f"[green]{speedup}[/green]"
            elif speedup_factor < 1.0:
                speedup = f"[red]{speedup} (slower!)[/red]"

        table.add_row(
            bench_name,
            f"{avg:.2f}",
            f"{min_time:.2f}",
            f"{max_time:.2f}",
            speedup
        )

    console.print(table)
    console.print()

    # Detailed breakdown
    for bench_name, bench_results in results.items():
        console.print(f"\n[bold]{bench_name} - Detailed Breakdown:[/bold]")
        detail_table = Table()
        detail_table.add_column("Iteration", style="cyan")
        detail_table.add_column("Restore (s)", justify="right")
        detail_table.add_column("Health (s)", justify="right")
        detail_table.add_column("Total (s)", justify="right")
        detail_table.add_column("Status", justify="center")

        for result in bench_results:
            status = "✅" if result["success"] else "❌"
            detail_table.add_row(
                str(result["iteration"]),
                f"{result['restore_time']:.2f}",
                f"{result['health_time']:.2f}",
                f"{result['total_time']:.2f}",
                status
            )

        console.print(detail_table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
