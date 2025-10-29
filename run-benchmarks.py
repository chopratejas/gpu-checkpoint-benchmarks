#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
# ]
# ///

"""
Benchmark Orchestration Script

Orchestrates the full GPU checkpoint benchmark suite:
1. Loads environment configuration
2. Ensures checkpoint exists (or creates it)
3. Runs benchmark iterations (baseline + CRIU)
4. Analyzes and displays results
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table


console = Console()


def load_env_file(env_path: Path = Path(".env")) -> dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars


def run_script(script_name: str, description: str, env: Optional[dict] = None) -> bool:
    """
    Run a Python script and return success status.

    Args:
        script_name: Name of the script to run
        description: Human-readable description for progress display
        env: Optional environment variables to pass to the script

    Returns:
        True if script succeeded, False otherwise
    """
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        console.print(f"[red]Error: Script {script_name} not found at {script_path}[/red]")
        return False

    console.print(f"\n[cyan]Running: {description}[/cyan]")

    # Merge current environment with provided env vars
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        result = subprocess.run(
            ["uv", "run", str(script_path)],
            env=run_env,
            check=False,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            console.print(f"[green]✓ {description} completed successfully[/green]")
            return True
        else:
            console.print(f"[red]✗ {description} failed with exit code {result.returncode}[/red]")
            return False

    except Exception as e:
        console.print(f"[red]✗ {description} failed with exception: {e}[/red]")
        return False


def check_checkpoint_exists(checkpoint_dir: str) -> bool:
    """Check if checkpoint directory exists and contains checkpoint files."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return False

    # Check for common CRIU checkpoint files
    required_files = ["inventory.img", "core-1.img"]
    return all((checkpoint_path / f).exists() for f in required_files)


def cooldown(seconds: int = 10):
    """Cooldown period between benchmarks with progress display."""
    console.print(f"[yellow]Cooling down for {seconds} seconds...[/yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Cooldown ({seconds}s)", total=seconds)
        for _ in range(seconds):
            time.sleep(1)
            progress.update(task, advance=1)


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate GPU checkpoint benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run 5 iterations of both benchmarks
  %(prog)s --iterations 10          # Run 10 iterations
  %(prog)s --baseline-only          # Only run baseline benchmarks
  %(prog)s --criu-only              # Only run CRIU benchmarks
  %(prog)s --skip-checkpoint        # Skip checkpoint creation check
        """
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        metavar="N",
        help="Number of benchmark iterations to run (default: 5)"
    )

    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Skip checkpoint creation/verification step"
    )

    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline benchmarks (no CRIU)"
    )

    parser.add_argument(
        "--criu-only",
        action="store_true",
        help="Only run CRIU benchmarks (no baseline)"
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.baseline_only and args.criu_only:
        console.print("[red]Error: --baseline-only and --criu-only are mutually exclusive[/red]")
        sys.exit(1)

    if args.iterations < 1:
        console.print("[red]Error: --iterations must be at least 1[/red]")
        sys.exit(1)

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]GPU Checkpoint Benchmark Suite[/bold cyan]\n"
        f"Iterations: {args.iterations}\n"
        f"Mode: {'Baseline Only' if args.baseline_only else 'CRIU Only' if args.criu_only else 'Both'}",
        border_style="cyan"
    ))

    # Step 1: Load environment
    console.print("\n[bold]Step 1: Loading environment configuration[/bold]")
    env_vars = load_env_file()

    if env_vars:
        console.print(f"[green]✓ Loaded {len(env_vars)} environment variables from .env[/green]")
    else:
        console.print("[yellow]⚠ No .env file found, using default configuration[/yellow]")

    checkpoint_dir = env_vars.get("CHECKPOINT_DIR", "/tmp/gpu-checkpoint")

    # Step 2: Check/create checkpoint
    if not args.skip_checkpoint and not args.baseline_only:
        console.print("\n[bold]Step 2: Verifying checkpoint[/bold]")

        if check_checkpoint_exists(checkpoint_dir):
            console.print(f"[green]✓ Checkpoint exists at {checkpoint_dir}[/green]")
        else:
            console.print(f"[yellow]⚠ Checkpoint not found at {checkpoint_dir}[/yellow]")
            console.print("[cyan]Creating checkpoint...[/cyan]")

            if not run_script("create-checkpoint.py", "Checkpoint Creation", env_vars):
                console.print("[red]Failed to create checkpoint. Exiting.[/red]")
                sys.exit(1)
    else:
        if args.skip_checkpoint:
            console.print("\n[bold]Step 2: Checkpoint verification skipped (--skip-checkpoint)[/bold]")
        else:
            console.print("\n[bold]Step 2: Checkpoint not needed for baseline-only mode[/bold]")

    # Step 3: Run benchmark iterations
    console.print(f"\n[bold]Step 3: Running {args.iterations} benchmark iteration(s)[/bold]")

    results = {
        "baseline_success": 0,
        "baseline_failed": 0,
        "criu_success": 0,
        "criu_failed": 0,
    }

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:

        task = progress.add_task(
            f"[cyan]Overall Progress",
            total=args.iterations
        )

        for i in range(1, args.iterations + 1):
            console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
            console.print(f"[bold magenta]Iteration {i}/{args.iterations}[/bold magenta]")
            console.print(f"[bold magenta]{'='*60}[/bold magenta]")

            # Run baseline benchmark
            if not args.criu_only:
                if run_script("benchmark-baseline.py", f"Baseline Benchmark (Iteration {i})", env_vars):
                    results["baseline_success"] += 1
                else:
                    results["baseline_failed"] += 1
                    console.print("[yellow]⚠ Baseline benchmark failed, continuing...[/yellow]")

                # Cooldown after baseline (unless it's baseline-only and last iteration)
                if not args.baseline_only or i < args.iterations:
                    cooldown(10)

            # Run CRIU benchmark
            if not args.baseline_only:
                if run_script("benchmark-criu.py", f"CRIU Benchmark (Iteration {i})", env_vars):
                    results["criu_success"] += 1
                else:
                    results["criu_failed"] += 1
                    console.print("[yellow]⚠ CRIU benchmark failed, continuing...[/yellow]")

                # Cooldown after CRIU (unless last iteration)
                if i < args.iterations:
                    cooldown(10)

            progress.update(task, advance=1)

    # Step 4: Analyze results
    console.print("\n[bold]Step 4: Analyzing results[/bold]")

    analysis_success = run_script("analyze-results.py", "Results Analysis", env_vars)

    if not analysis_success:
        console.print("[yellow]⚠ Results analysis failed or not available[/yellow]")

    # Step 5: Display summary
    console.print("\n[bold]Step 5: Summary[/bold]")

    # Create results table
    table = Table(title="Benchmark Results Summary", border_style="cyan")
    table.add_column("Benchmark Type", style="cyan", no_wrap=True)
    table.add_column("Successful", style="green", justify="right")
    table.add_column("Failed", style="red", justify="right")
    table.add_column("Total", style="yellow", justify="right")

    if not args.criu_only:
        baseline_total = results["baseline_success"] + results["baseline_failed"]
        table.add_row(
            "Baseline",
            str(results["baseline_success"]),
            str(results["baseline_failed"]),
            str(baseline_total)
        )

    if not args.baseline_only:
        criu_total = results["criu_success"] + results["criu_failed"]
        table.add_row(
            "CRIU",
            str(results["criu_success"]),
            str(results["criu_failed"]),
            str(criu_total)
        )

    console.print(table)

    # Display results location
    results_dir = Path("results")
    if results_dir.exists():
        console.print(f"\n[green]✓ Results saved to: {results_dir.absolute()}[/green]")

        # List result files
        result_files = list(results_dir.glob("*.csv")) + list(results_dir.glob("*.json"))
        if result_files:
            console.print("\n[cyan]Result files:[/cyan]")
            for f in sorted(result_files)[-5:]:  # Show last 5 files
                console.print(f"  • {f.name}")
    else:
        console.print(f"\n[yellow]⚠ Results directory not found at {results_dir.absolute()}[/yellow]")

    # Determine exit status
    total_failed = results["baseline_failed"] + results["criu_failed"]

    if total_failed == 0:
        console.print("\n[bold green]✓ All benchmarks completed successfully![/bold green]")
        sys.exit(0)
    else:
        console.print(f"\n[bold yellow]⚠ Completed with {total_failed} failed benchmark(s)[/bold yellow]")
        sys.exit(0)  # Still exit 0 since we handled errors gracefully


if __name__ == "__main__":
    main()
