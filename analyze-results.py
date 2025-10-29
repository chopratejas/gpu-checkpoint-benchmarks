#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich>=13.7.0",
#     "numpy>=1.26.0",
# ]
# ///

"""
Statistical Analysis Tool for GPU Checkpoint Benchmark Results

This script analyzes benchmark results comparing baseline (cold start) vs CRIU
(checkpoint/restore) performance for GPU workloads.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

console = Console()

# Configuration
RESULTS_DIR = Path("/root/gpu-load/results")


@dataclass
class Statistics:
    """Statistical metrics for a set of measurements."""
    mean: float
    median: float
    stddev: float
    min: float
    max: float
    count: int

    def __str__(self) -> str:
        return f"{self.mean:.3f}s (±{self.stddev:.3f}s)"


@dataclass
class BenchmarkData:
    """Container for benchmark results."""
    baseline_runs: List[Dict] = field(default_factory=list)
    criu_runs: List[Dict] = field(default_factory=list)
    checkpoint_info: Optional[Dict] = None


def calculate_stats(values: List[float]) -> Optional[Statistics]:
    """Calculate statistical metrics for a list of values."""
    if not values:
        return None

    arr = np.array(values)
    return Statistics(
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        stddev=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        count=len(arr)
    )


def load_results() -> BenchmarkData:
    """Load all JSON result files from RESULTS_DIR."""
    if not RESULTS_DIR.exists():
        console.print(f"[red]Error: Results directory not found: {RESULTS_DIR}[/red]")
        sys.exit(1)

    data = BenchmarkData()
    json_files = list(RESULTS_DIR.glob("*.json"))

    if not json_files:
        console.print(f"[red]Error: No JSON files found in {RESULTS_DIR}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading results from {RESULTS_DIR}...[/cyan]")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)

            # Check if this is a checkpoint info file
            if "checkpoint_path" in result and "checkpoint_size_mb" in result:
                if data.checkpoint_info is None:
                    data.checkpoint_info = result
                continue

            # Classify as baseline or CRIU run
            if result.get("mode") == "baseline":
                data.baseline_runs.append(result)
            elif result.get("mode") == "criu":
                data.criu_runs.append(result)
            else:
                console.print(f"[yellow]Warning: Unknown mode in {json_file.name}[/yellow]")

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {json_file.name}: {e}[/yellow]")

    console.print(f"[green]Loaded {len(data.baseline_runs)} baseline runs and {len(data.criu_runs)} CRIU runs[/green]")
    return data


def extract_metrics(runs: List[Dict], mode: str) -> Dict[str, List[float]]:
    """Extract timing metrics from runs."""
    metrics = {
        "total_tfft": [],
        "inference_tfft": [],
    }

    if mode == "baseline":
        metrics.update({
            "container_start": [],
            "model_load": [],
        })
    elif mode == "criu":
        metrics.update({
            "restore_time": [],
            "reinit_time": [],
        })

    for run in runs:
        timings = run.get("timings", {})

        # Common metrics
        if "total_tfft" in timings:
            metrics["total_tfft"].append(timings["total_tfft"])
        if "inference_tfft" in timings:
            metrics["inference_tfft"].append(timings["inference_tfft"])

        # Mode-specific metrics
        if mode == "baseline":
            if "container_start" in timings:
                metrics["container_start"].append(timings["container_start"])
            if "model_load" in timings:
                metrics["model_load"].append(timings["model_load"])
        elif mode == "criu":
            if "restore_time" in timings:
                metrics["restore_time"].append(timings["restore_time"])
            if "reinit_time" in timings:
                metrics["reinit_time"].append(timings["reinit_time"])

    return metrics


def create_summary_table(baseline_stats: Dict, criu_stats: Dict) -> Table:
    """Create a summary comparison table."""
    table = Table(
        title="Benchmark Summary: Baseline vs CRIU",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Baseline", justify="right", style="yellow")
    table.add_column("CRIU", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold blue")

    # Total TFFT comparison
    if baseline_stats.get("total_tfft") and criu_stats.get("total_tfft"):
        baseline_tfft = baseline_stats["total_tfft"]
        criu_tfft = criu_stats["total_tfft"]
        speedup = baseline_tfft.mean / criu_tfft.mean
        speedup_pct = (speedup - 1) * 100

        table.add_row(
            "Total TFFT",
            f"{baseline_tfft.mean:.3f}s ±{baseline_tfft.stddev:.3f}s",
            f"{criu_tfft.mean:.3f}s ±{criu_tfft.stddev:.3f}s",
            f"{speedup:.2f}x ({speedup_pct:+.1f}%)"
        )

    # Inference TFFT comparison
    if baseline_stats.get("inference_tfft") and criu_stats.get("inference_tfft"):
        baseline_inf = baseline_stats["inference_tfft"]
        criu_inf = criu_stats["inference_tfft"]

        table.add_row(
            "Inference TFFT",
            f"{baseline_inf.mean:.3f}s ±{baseline_inf.stddev:.3f}s",
            f"{criu_inf.mean:.3f}s ±{criu_inf.stddev:.3f}s",
            "~1.0x (similar)"
        )

    return table


def create_detailed_table(stats: Dict, title: str, color: str) -> Table:
    """Create a detailed statistics table for a single mode."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style=f"bold {color}"
    )

    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("N", justify="right")

    for metric_name, stat in stats.items():
        if stat is None:
            continue

        # Format metric name
        display_name = metric_name.replace("_", " ").title()

        table.add_row(
            display_name,
            f"{stat.mean:.3f}s",
            f"{stat.median:.3f}s",
            f"{stat.stddev:.3f}s",
            f"{stat.min:.3f}s",
            f"{stat.max:.3f}s",
            str(stat.count)
        )

    return table


def create_ascii_chart(baseline_stats: Dict, criu_stats: Dict, width: int = 60) -> str:
    """Create a simple ASCII bar chart showing timing breakdown."""
    chart_lines = []
    chart_lines.append("Timing Breakdown Comparison")
    chart_lines.append("=" * width)
    chart_lines.append("")

    # Get baseline breakdown
    baseline_total = baseline_stats.get("total_tfft")
    if baseline_total:
        chart_lines.append("BASELINE (Cold Start):")
        chart_lines.append("-" * width)

        container = baseline_stats.get("container_start")
        model = baseline_stats.get("model_load")
        inference = baseline_stats.get("inference_tfft")

        if container:
            pct = (container.mean / baseline_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Container Start: {'█' * bar_len} {container.mean:.3f}s ({pct:.1f}%)")

        if model:
            pct = (model.mean / baseline_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Model Load:      {'█' * bar_len} {model.mean:.3f}s ({pct:.1f}%)")

        if inference:
            pct = (inference.mean / baseline_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Inference TFFT:  {'█' * bar_len} {inference.mean:.3f}s ({pct:.1f}%)")

        chart_lines.append(f"  TOTAL:           {baseline_total.mean:.3f}s")
        chart_lines.append("")

    # Get CRIU breakdown
    criu_total = criu_stats.get("total_tfft")
    if criu_total:
        chart_lines.append("CRIU (Checkpoint/Restore):")
        chart_lines.append("-" * width)

        restore = criu_stats.get("restore_time")
        reinit = criu_stats.get("reinit_time")
        inference = criu_stats.get("inference_tfft")

        if restore:
            pct = (restore.mean / criu_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Restore Time:    {'█' * bar_len} {restore.mean:.3f}s ({pct:.1f}%)")

        if reinit:
            pct = (reinit.mean / criu_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Reinit Time:     {'█' * bar_len} {reinit.mean:.3f}s ({pct:.1f}%)")

        if inference:
            pct = (inference.mean / criu_total.mean) * 100
            bar_len = int((pct / 100) * (width - 30))
            chart_lines.append(f"  Inference TFFT:  {'█' * bar_len} {inference.mean:.3f}s ({pct:.1f}%)")

        chart_lines.append(f"  TOTAL:           {criu_total.mean:.3f}s")
        chart_lines.append("")

    return "\n".join(chart_lines)


def generate_markdown_report(
    baseline_stats: Dict,
    criu_stats: Dict,
    checkpoint_info: Optional[Dict],
    ascii_chart: str
) -> str:
    """Generate a markdown report."""
    lines = []
    lines.append("# GPU Checkpoint Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## Summary\n")

    # Overall comparison
    baseline_tfft = baseline_stats.get("total_tfft")
    criu_tfft = criu_stats.get("total_tfft")

    if baseline_tfft and criu_tfft:
        speedup = baseline_tfft.mean / criu_tfft.mean
        speedup_pct = (speedup - 1) * 100
        time_saved = baseline_tfft.mean - criu_tfft.mean

        lines.append(f"- **Baseline TFFT**: {baseline_tfft.mean:.3f}s (±{baseline_tfft.stddev:.3f}s)")
        lines.append(f"- **CRIU TFFT**: {criu_tfft.mean:.3f}s (±{criu_tfft.stddev:.3f}s)")
        lines.append(f"- **Speedup**: {speedup:.2f}x ({speedup_pct:+.1f}%)")
        lines.append(f"- **Time Saved**: {time_saved:.3f}s per inference")
        lines.append("")

    # Checkpoint information
    if checkpoint_info:
        lines.append("## Checkpoint Information\n")
        lines.append(f"- **Path**: `{checkpoint_info.get('checkpoint_path', 'N/A')}`")
        lines.append(f"- **Size**: {checkpoint_info.get('checkpoint_size_mb', 'N/A')} MB")
        if "creation_time" in checkpoint_info:
            lines.append(f"- **Creation Time**: {checkpoint_info['creation_time']:.3f}s")
        lines.append("")

    # Detailed statistics - Baseline
    lines.append("## Baseline (Cold Start) Statistics\n")
    lines.append("| Metric | Mean | Median | Std Dev | Min | Max | N |")
    lines.append("|--------|------|--------|---------|-----|-----|---|")

    for metric_name, stat in baseline_stats.items():
        if stat is None:
            continue
        display_name = metric_name.replace("_", " ").title()
        lines.append(
            f"| {display_name} | {stat.mean:.3f}s | {stat.median:.3f}s | "
            f"{stat.stddev:.3f}s | {stat.min:.3f}s | {stat.max:.3f}s | {stat.count} |"
        )

    lines.append("")

    # Detailed statistics - CRIU
    lines.append("## CRIU (Checkpoint/Restore) Statistics\n")
    lines.append("| Metric | Mean | Median | Std Dev | Min | Max | N |")
    lines.append("|--------|------|--------|---------|-----|-----|---|")

    for metric_name, stat in criu_stats.items():
        if stat is None:
            continue
        display_name = metric_name.replace("_", " ").title()
        lines.append(
            f"| {display_name} | {stat.mean:.3f}s | {stat.median:.3f}s | "
            f"{stat.stddev:.3f}s | {stat.min:.3f}s | {stat.max:.3f}s | {stat.count} |"
        )

    lines.append("")

    # ASCII chart
    lines.append("## Timing Breakdown\n")
    lines.append("```")
    lines.append(ascii_chart)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    console.print(Panel.fit(
        "[bold cyan]GPU Checkpoint Benchmark Analysis[/bold cyan]",
        border_style="cyan"
    ))

    # Load data
    data = load_results()

    if not data.baseline_runs:
        console.print("[red]Error: No baseline runs found[/red]")
        sys.exit(1)

    if not data.criu_runs:
        console.print("[red]Error: No CRIU runs found[/red]")
        sys.exit(1)

    # Extract metrics
    baseline_metrics = extract_metrics(data.baseline_runs, "baseline")
    criu_metrics = extract_metrics(data.criu_runs, "criu")

    # Calculate statistics
    baseline_stats = {
        metric: calculate_stats(values)
        for metric, values in baseline_metrics.items()
    }

    criu_stats = {
        metric: calculate_stats(values)
        for metric, values in criu_metrics.items()
    }

    # Display summary table
    console.print()
    summary_table = create_summary_table(baseline_stats, criu_stats)
    console.print(summary_table)
    console.print()

    # Display detailed tables
    baseline_table = create_detailed_table(
        baseline_stats,
        "Baseline (Cold Start) - Detailed Statistics",
        "yellow"
    )
    console.print(baseline_table)
    console.print()

    criu_table = create_detailed_table(
        criu_stats,
        "CRIU (Checkpoint/Restore) - Detailed Statistics",
        "green"
    )
    console.print(criu_table)
    console.print()

    # Display checkpoint info if available
    if data.checkpoint_info:
        info_table = Table(
            title="Checkpoint Information",
            box=box.ROUNDED,
            show_header=False,
            border_style="blue"
        )
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Path", data.checkpoint_info.get("checkpoint_path", "N/A"))
        info_table.add_row("Size", f"{data.checkpoint_info.get('checkpoint_size_mb', 'N/A')} MB")
        if "creation_time" in data.checkpoint_info:
            info_table.add_row("Creation Time", f"{data.checkpoint_info['creation_time']:.3f}s")

        console.print(info_table)
        console.print()

    # Generate ASCII chart
    ascii_chart = create_ascii_chart(baseline_stats, criu_stats)
    console.print(Panel(ascii_chart, title="Timing Breakdown", border_style="blue"))
    console.print()

    # Generate markdown report
    markdown = generate_markdown_report(
        baseline_stats,
        criu_stats,
        data.checkpoint_info,
        ascii_chart
    )

    # Save report
    report_path = RESULTS_DIR / "benchmark_report.md"
    report_path.write_text(markdown)

    console.print(f"[bold green]Report saved to: {report_path}[/bold green]")

    # Final summary
    if baseline_stats.get("total_tfft") and criu_stats.get("total_tfft"):
        baseline_tfft = baseline_stats["total_tfft"]
        criu_tfft = criu_stats["total_tfft"]
        speedup = baseline_tfft.mean / criu_tfft.mean
        speedup_pct = (speedup - 1) * 100

        summary_text = Text()
        summary_text.append("\nKey Findings:\n", style="bold white")
        summary_text.append(f"  Baseline TFFT: ", style="yellow")
        summary_text.append(f"{baseline_tfft.mean:.3f}s\n", style="bold yellow")
        summary_text.append(f"  CRIU TFFT:     ", style="green")
        summary_text.append(f"{criu_tfft.mean:.3f}s\n", style="bold green")
        summary_text.append(f"  Speedup:       ", style="blue")
        summary_text.append(f"{speedup:.2f}x ({speedup_pct:+.1f}%)\n", style="bold blue")

        console.print(Panel(summary_text, border_style="magenta"))


if __name__ == "__main__":
    main()
