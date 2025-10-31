#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
# ]
# ///

"""Simple benchmark comparing system CRIU vs custom CRIU."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

import utils

console = Console()


def swap_criu_binary(criu_path: str) -> None:
    """Swap the system CRIU binary."""
    console.print(f"[cyan]Installing CRIU: {criu_path}[/cyan]")
    shutil.copy(criu_path, "/usr/sbin/criu")
    result = subprocess.run(
        ["criu", "--version"],
        capture_output=True,
        text=True
    )
    console.print(f"[dim]{result.stdout.strip()}[/dim]")


def stop_container(container_name: str) -> None:
    """Stop container if running."""
    subprocess.run(
        ["podman", "stop", container_name],
        capture_output=True,
        check=False
    )
    time.sleep(1)


def restore_container(container_name: str) -> tuple[float, bool]:
    """
    Restore container and measure time.

    Returns:
        Tuple of (restore_time_seconds, success)
    """
    start_time = time.time()

    result = subprocess.run(
        ["podman", "container", "restore", "--keep", "--print-stats", container_name],
        capture_output=True,
        text=True,
        check=False
    )

    restore_time = time.time() - start_time

    if result.returncode != 0:
        console.print(f"[red]Restore failed![/red]")
        console.print(f"[red]Error: {result.stderr}[/red]")
        return restore_time, False

    console.print(f"[green]Restore completed in {restore_time:.3f}s[/green]")
    return restore_time, True


def verify_container_healthy(port: int = 8000, timeout: int = 60) -> tuple[float, bool]:
    """Check if container is healthy and accepting requests."""
    console.print(f"[cyan]Checking health...[/cyan]")

    try:
        health_time = utils.wait_for_health(port=port, timeout=timeout)
        console.print(f"[green]Healthy in {health_time:.3f}s[/green]")
        return health_time, True
    except TimeoutError as e:
        console.print(f"[red]Health check failed: {e}[/red]")
        return 0.0, False


def run_iteration(
    criu_name: str,
    container_name: str,
    api_port: int,
    iteration: int
) -> dict:
    """Run single benchmark iteration."""
    console.print(f"[bold]Iteration {iteration}[/bold]")

    # 1. Stop container if running
    stop_container(container_name)

    # 2. Restore
    restore_time, restore_success = restore_container(container_name)

    if not restore_success:
        return {
            "iteration": iteration,
            "restore_time": restore_time,
            "health_time": 0.0,
            "total_time": restore_time,
            "success": False
        }

    # 3. Wait for health
    health_time, health_success = verify_container_healthy(port=api_port, timeout=120)

    total_time = restore_time + health_time

    return {
        "iteration": iteration,
        "restore_time": restore_time,
        "health_time": health_time,
        "total_time": total_time,
        "success": health_success
    }


def main() -> int:
    """Main benchmark function."""
    console.print("[bold blue]CRIU Comparison Benchmark[/bold blue]\n")

    # Configuration
    env_vars = utils.load_env()
    if not env_vars:
        console.print("[red]Failed to load .env file[/red]")
        return 1

    container_name = env_vars.get("CONT_NAME", "vllm-llm-demo")
    api_port = int(env_vars.get("API_PORT", "8000"))
    iterations = 3

    # Check binaries exist
    system_criu = Path("/usr/sbin/criu.backup")
    custom_criu = Path("/root/criu/criu/criu")

    if not system_criu.exists():
        console.print(f"[red]System CRIU backup not found: {system_criu}[/red]")
        return 1

    if not custom_criu.exists():
        console.print(f"[red]Custom CRIU not found: {custom_criu}[/red]")
        return 1

    # Check container has checkpoint
    result = subprocess.run(
        ["podman", "inspect", "--format", "{{.Id}}", container_name],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        console.print(f"[red]Container {container_name} not found![/red]")
        return 1

    container_id = result.stdout.strip()
    console.print(f"[green]Container: {container_id[:12]}[/green]")

    ckpt_dir = Path(f"/var/lib/containers/storage/overlay-containers/{container_id}/userdata/checkpoint")
    if not ckpt_dir.exists():
        console.print(f"[red]No checkpoint found![/red]")
        console.print("[yellow]Run ./create-checkpoint.py first[/yellow]")
        return 1

    console.print(f"[green]Checkpoint: {ckpt_dir}[/green]\n")

    # Benchmarks
    benchmarks = [
        {
            "name": "System CRIU v4.1.1 (baseline)",
            "criu_path": str(system_criu),
        },
        {
            "name": "Custom CRIU v4.0 (with readahead)",
            "criu_path": str(custom_criu),
        },
    ]

    results = {}

    # Run benchmarks
    for benchmark in benchmarks:
        bench_name = benchmark["name"]
        criu_path = benchmark["criu_path"]

        console.print(f"\n{'='*70}")
        console.print(f"[bold green]{bench_name}[/bold green]")
        console.print(f"{'='*70}\n")

        # Install CRIU binary
        swap_criu_binary(criu_path)

        bench_results = []

        for i in range(1, iterations + 1):
            result = run_iteration(
                criu_name=bench_name,
                container_name=container_name,
                api_port=api_port,
                iteration=i
            )
            bench_results.append(result)
            time.sleep(2)  # Brief pause

        results[bench_name] = bench_results

    # Display results
    console.print(f"\n{'='*70}")
    console.print("[bold green]Results Summary[/bold green]")
    console.print(f"{'='*70}\n")

    table = Table(title="CRIU Restore Performance Comparison")
    table.add_column("CRIU", style="cyan")
    table.add_column("Avg Restore (s)", justify="right", style="green")
    table.add_column("Avg Health (s)", justify="right")
    table.add_column("Avg Total (s)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right", style="bold")

    baseline_restore = None

    for bench_name, bench_results in results.items():
        successful = [r for r in bench_results if r["success"]]
        if not successful:
            table.add_row(bench_name, "FAILED", "-", "-", "-")
            continue

        restore_times = [r["restore_time"] for r in successful]
        health_times = [r["health_time"] for r in successful]
        total_times = [r["total_time"] for r in successful]

        avg_restore = sum(restore_times) / len(restore_times)
        avg_health = sum(health_times) / len(health_times)
        avg_total = sum(total_times) / len(total_times)

        if baseline_restore is None:
            baseline_restore = avg_restore
            speedup = "1.00x (baseline)"
        else:
            speedup_factor = baseline_restore / avg_restore
            improvement = ((baseline_restore - avg_restore) / baseline_restore) * 100
            speedup = f"{speedup_factor:.2f}x ({improvement:+.1f}%)"

        table.add_row(
            bench_name,
            f"{avg_restore:.3f}",
            f"{avg_health:.3f}",
            f"{avg_total:.3f}",
            speedup
        )

    console.print(table)
    console.print()

    # Detailed breakdown
    for bench_name, bench_results in results.items():
        console.print(f"\n[bold]{bench_name} - Details:[/bold]")
        detail_table = Table()
        detail_table.add_column("Iter", style="cyan")
        detail_table.add_column("Restore (s)", justify="right")
        detail_table.add_column("Health (s)", justify="right")
        detail_table.add_column("Total (s)", justify="right")
        detail_table.add_column("Status", justify="center")

        for result in bench_results:
            status = "✅" if result["success"] else "❌"
            detail_table.add_row(
                str(result["iteration"]),
                f"{result['restore_time']:.3f}",
                f"{result['health_time']:.3f}",
                f"{result['total_time']:.3f}",
                status
            )

        console.print(detail_table)

    console.print("\n[bold]Note:[/bold] Persistence mode enabled for fair comparison")

    return 0


if __name__ == "__main__":
    sys.exit(main())
