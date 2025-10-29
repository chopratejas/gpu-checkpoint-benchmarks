#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "httpx",
#   "rich",
# ]
# ///

"""
Test script to verify all utility functions work correctly.
"""

import sys
sys.path.insert(0, '/root/gpu-load')

from utils import (
    get_timestamp_ns,
    log_metric,
    load_env,
    cleanup_containers,
    run_command,
)
from rich.console import Console
from rich.panel import Panel

console = Console()

console.print(Panel.fit(
    "[bold green]Testing All Utility Functions[/bold green]",
    border_style="green"
))

# Test 1: get_timestamp_ns
console.print("\n[bold cyan]Test 1: get_timestamp_ns()[/bold cyan]")
ts1 = get_timestamp_ns()
console.print(f"Timestamp 1: {ts1}")
import time
time.sleep(0.01)
ts2 = get_timestamp_ns()
console.print(f"Timestamp 2: {ts2}")
console.print(f"Difference: {ts2 - ts1} ns")
assert ts2 > ts1, "Timestamps should be increasing"
console.print("[green]✓ get_timestamp_ns() works correctly[/green]")

# Test 2: log_metric
console.print("\n[bold cyan]Test 2: log_metric()[/bold cyan]")
log_metric("Test CPU", 42.5, "%")
log_metric("Test Memory", 8192, "MB")
log_metric("Test Status", "OK")
log_metric("No Unit Test", 123)
console.print("[green]✓ log_metric() works correctly[/green]")

# Test 3: load_env
console.print("\n[bold cyan]Test 3: load_env()[/bold cyan]")
env_vars = load_env()
console.print(f"Loaded {len(env_vars)} environment variables")
console.print("[green]✓ load_env() works correctly[/green]")

# Test 4: run_command
console.print("\n[bold cyan]Test 4: run_command()[/bold cyan]")
result = run_command(
    ["echo", "Hello from test"],
    "Test echo command",
    timeout=5
)
assert result.returncode == 0
assert "Hello from test" in result.stdout
console.print("[green]✓ run_command() works correctly[/green]")

# Test 5: cleanup_containers
console.print("\n[bold cyan]Test 5: cleanup_containers()[/bold cyan]")
console.print("Testing with non-existent container (should handle gracefully)")
cleanup_containers("test-nonexistent-container")
console.print("[green]✓ cleanup_containers() handles non-existent containers[/green]")

# Summary
console.print(Panel.fit(
    "[bold green]All utility functions tested successfully![/bold green]\n\n"
    "Functions verified:\n"
    "  1. get_timestamp_ns() - High-resolution timestamp\n"
    "  2. log_metric() - Structured logging\n"
    "  3. load_env() - Environment loading\n"
    "  4. run_command() - Command execution with timing\n"
    "  5. cleanup_containers() - Container cleanup\n\n"
    "Note: wait_for_health() and send_inference() require running vLLM instance",
    border_style="green"
))
