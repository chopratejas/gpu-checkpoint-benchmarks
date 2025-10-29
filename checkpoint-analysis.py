#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
# ]
# ///

"""Analyze CRIU checkpoint contents and size breakdown.

This script inspects a CRIU checkpoint tar archive to understand:
- Total size and compression ratio
- Memory pages breakdown (CPU vs GPU)
- File contents and their sizes
- GPU memory allocation details

Helps diagnose why checkpoints are large and where optimization opportunities exist.
"""

import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_checkpoint_path() -> Path:
    """Get checkpoint path from command line or default location."""
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    else:
        ckpt_path = Path("/mnt/checkpoint-ram/checkpoint.tar")

    if not ckpt_path.exists():
        console.print(f"[red]Error: Checkpoint not found at {ckpt_path}[/red]")
        console.print(f"[yellow]Usage: {sys.argv[0]} [checkpoint_path][/yellow]")
        sys.exit(1)

    return ckpt_path


def format_size(bytes_size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def analyze_tar_contents(ckpt_path: Path) -> Tuple[int, Dict[str, int], List[Tuple[str, int]]]:
    """
    Analyze tar archive contents.

    Returns:
        Tuple of (total_size, category_sizes, all_files)
    """
    console.print(f"[cyan]Analyzing checkpoint: {ckpt_path}[/cyan]")

    category_sizes = {
        "memory_pages": 0,
        "gpu_data": 0,
        "process_state": 0,
        "filesystem": 0,
        "network": 0,
        "other": 0,
    }

    all_files = []
    total_size = 0

    try:
        with tarfile.open(ckpt_path, 'r') as tar:
            members = tar.getmembers()

            for member in members:
                if not member.isfile():
                    continue

                size = member.size
                name = member.name
                total_size += size
                all_files.append((name, size))

                # Categorize files
                if 'pages-' in name or 'pagemap-' in name:
                    category_sizes["memory_pages"] += size
                elif 'gpu' in name.lower() or 'cuda' in name.lower():
                    category_sizes["gpu_data"] += size
                elif any(x in name for x in ['core-', 'mm-', 'pstree', 'ids-', 'tty-']):
                    category_sizes["process_state"] += size
                elif 'fs-' in name or 'mountpoints' in name:
                    category_sizes["filesystem"] += size
                elif any(x in name for x in ['netdev-', 'ifaddr-', 'route-']):
                    category_sizes["network"] += size
                else:
                    category_sizes["other"] += size

    except Exception as e:
        console.print(f"[red]Error analyzing tar: {e}[/red]")
        sys.exit(1)

    return total_size, category_sizes, all_files


def display_summary(ckpt_path: Path, total_size: int, category_sizes: Dict[str, int]):
    """Display summary statistics."""
    disk_size = ckpt_path.stat().st_size

    # Create summary table
    table = Table(title="Checkpoint Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Checkpoint Path", str(ckpt_path))
    table.add_row("Disk Size", format_size(disk_size))
    table.add_row("Uncompressed Size", format_size(total_size))

    if disk_size > 0:
        compression_ratio = total_size / disk_size
        table.add_row("Compression Ratio", f"{compression_ratio:.2f}x")

    console.print(table)


def display_category_breakdown(category_sizes: Dict[str, int], total_size: int):
    """Display breakdown by category."""
    table = Table(title="Size Breakdown by Category", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Percentage", style="yellow")

    # Sort by size descending
    sorted_categories = sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)

    for category, size in sorted_categories:
        if size > 0:
            percentage = (size / total_size * 100) if total_size > 0 else 0
            table.add_row(
                category.replace("_", " ").title(),
                format_size(size),
                f"{percentage:.1f}%"
            )

    console.print(table)


def display_largest_files(all_files: List[Tuple[str, int]], n: int = 20):
    """Display largest files in checkpoint."""
    table = Table(title=f"Top {n} Largest Files", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim")
    table.add_column("File Name", style="cyan")
    table.add_column("Size", style="green")

    # Sort by size descending
    sorted_files = sorted(all_files, key=lambda x: x[1], reverse=True)

    for i, (name, size) in enumerate(sorted_files[:n], 1):
        table.add_row(str(i), name, format_size(size))

    console.print(table)


def analyze_criu_stats(ckpt_path: Path):
    """Try to extract and display CRIU statistics if available."""
    console.print("\n[bold]CRIU Statistics (if available):[/bold]")

    try:
        # Extract stats-dump file if it exists
        with tarfile.open(ckpt_path, 'r') as tar:
            try:
                stats_member = tar.getmember('stats-dump')
                stats_file = tar.extractfile(stats_member)
                if stats_file:
                    # Try to display as text
                    content = stats_file.read().decode('utf-8', errors='ignore')
                    console.print(Panel(content[:500], title="stats-dump (first 500 chars)"))
            except KeyError:
                console.print("[yellow]No stats-dump file found in checkpoint[/yellow]")

    except Exception as e:
        console.print(f"[yellow]Could not extract CRIU stats: {e}[/yellow]")


def display_optimization_hints(category_sizes: Dict[str, int], total_size: int):
    """Display optimization hints based on analysis."""
    console.print("\n[bold magenta]Optimization Hints:[/bold magenta]")

    hints = []

    # Check memory pages
    mem_pages_pct = (category_sizes["memory_pages"] / total_size * 100) if total_size > 0 else 0
    if mem_pages_pct > 50:
        hints.append(
            f"• Memory pages are {mem_pages_pct:.1f}% of checkpoint - consider using "
            "--load-format safetensors for file-backed memory"
        )

    # Check GPU data
    gpu_data_pct = (category_sizes["gpu_data"] / total_size * 100) if total_size > 0 else 0
    if gpu_data_pct > 30:
        hints.append(
            f"• GPU data is {gpu_data_pct:.1f}% of checkpoint - this is expected for vLLM. "
            "Ensure using CRIU 4.2+ for parallel GPU restore"
        )

    # Check filesystem
    fs_pct = (category_sizes["filesystem"] / total_size * 100) if total_size > 0 else 0
    if fs_pct > 5:
        hints.append(
            f"• Filesystem data is {fs_pct:.1f}% of checkpoint - use --ignore-rootfs "
            "to exclude unnecessary filesystem changes"
        )

    # Check compression
    if len(hints) == 0:
        hints.append("✓ Checkpoint looks well-optimized!")

    for hint in hints:
        console.print(f"[yellow]{hint}[/yellow]")


def main():
    """Main execution."""
    console.rule("[bold blue]CRIU Checkpoint Analysis")

    # Get checkpoint path
    ckpt_path = get_checkpoint_path()

    # Analyze contents
    total_size, category_sizes, all_files = analyze_tar_contents(ckpt_path)

    # Display results
    console.print()
    display_summary(ckpt_path, total_size, category_sizes)

    console.print()
    display_category_breakdown(category_sizes, total_size)

    console.print()
    display_largest_files(all_files, n=20)

    # Try to show CRIU stats
    analyze_criu_stats(ckpt_path)

    # Show optimization hints
    display_optimization_hints(category_sizes, total_size)

    console.rule("[bold green]Analysis Complete")


if __name__ == "__main__":
    main()
