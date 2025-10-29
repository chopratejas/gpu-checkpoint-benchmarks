#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx",
#     "rich",
#     "python-dotenv",
# ]
# ///

"""
benchmark-baseline.py - Traditional Cold Start TFFT Measurement

This script measures the baseline cold start performance of vLLM:
- T0: Start timer before podman run
- T1: Container started (podman run returns)
- T2: API ready (health check passes)
- T3: First token received from inference
- T4: Full response received

Metrics logged:
- container_start_time (T1-T0): Time to start container
- model_load_time (T2-T1): Time to load model and become ready
- inference_tfft (T3-T2): Time to first token after API ready
- total_cold_start_tfft (T3-T0): Total time from start to first token
- full_response_time (T4-T0): Total time to complete response
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def load_environment() -> dict[str, str]:
    """Load environment variables from .env file."""
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    if not env_file.exists():
        console.print(f"[red]Error: .env file not found at {env_file}[/red]")
        console.print("[yellow]Please run setup.sh first to create the environment[/yellow]")
        sys.exit(1)

    load_dotenv(env_file)

    # Load required environment variables
    config = {
        "MODEL_ID": os.getenv("MODEL_ID", "Qwen/Qwen2-1.5B-Instruct"),
        "CONT_NAME": os.getenv("CONT_NAME", "vllm-llm-demo"),
        "API_PORT": os.getenv("API_PORT", "8000"),
        "VLLM_IMAGE": os.getenv("VLLM_IMAGE", "docker.io/vllm/vllm-openai:latest"),
        "GPU_MEMORY_UTIL": os.getenv("GPU_MEMORY_UTIL", "0.90"),
        "MAX_MODEL_LEN": os.getenv("MAX_MODEL_LEN", "4096"),
        "NVIDIA_LIBS_PATH": os.getenv("NVIDIA_LIBS_PATH", "/opt/nvidia-libs"),
        "MODELS_CACHE": os.getenv("MODELS_CACHE", "/models"),
        "RESULTS_DIR": os.getenv("RESULTS_DIR", str(script_dir / "results")),
        "HEALTH_CHECK_TIMEOUT": os.getenv("HEALTH_CHECK_TIMEOUT", "300"),
        "HEALTH_CHECK_INTERVAL": os.getenv("HEALTH_CHECK_INTERVAL", "2"),
    }

    return config


def cleanup_containers(container_name: str) -> None:
    """Stop and remove any existing containers with the given name."""
    console.print(f"[cyan]Cleaning up existing containers: {container_name}[/cyan]")

    # Stop container if running
    try:
        subprocess.run(
            ["podman", "stop", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        console.print(f"[yellow]Warning: Timeout stopping container, forcing...[/yellow]")
        subprocess.run(
            ["podman", "kill", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # Container may not exist

    # Remove container
    try:
        subprocess.run(
            ["podman", "rm", "-f", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except Exception:
        pass  # Container may not exist


def cleanup_for_true_cold_start() -> None:
    """
    Complete cleanup to ensure TRUE cold start conditions.

    This eliminates all caching layers that could pollute measurements:
    - GPU memory (via process termination)
    - Linux page cache (model file contents)
    - Dentry/inode cache (filesystem metadata)
    - CUDA contexts (via process termination)

    Research shows page cache can make "cold" start 20x+ faster on repeat runs!
    """
    console.print("[bold cyan]Performing TRUE cold start cleanup...[/bold cyan]")

    # 1. Kill any stray GPU processes
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [p.strip() for p in result.stdout.strip().split('\n') if p.strip()]

        if pids:
            console.print(f"[yellow]Found {len(pids)} GPU processes, killing...[/yellow]")
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], timeout=2, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            time.sleep(2)
            console.print("[green]✓ GPU processes cleaned[/green]")
        else:
            console.print("[green]✓ No GPU processes found[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not check GPU processes: {e}[/yellow]")

    # 2. Drop Linux caches (CRITICAL for true cold start)
    console.print("[cyan]Dropping system caches (requires sudo)...[/cyan]")
    try:
        # Flush dirty pages to disk first
        subprocess.run(["sudo", "sync"], timeout=10, check=True, stderr=subprocess.DEVNULL)

        # Drop all caches: page cache + dentries + inodes
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            timeout=10,
            check=True,
            stderr=subprocess.DEVNULL
        )
        console.print("[green]✓ System caches dropped (page cache, dentries, inodes)[/green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]ERROR: Could not drop caches (need sudo)[/red]")
        console.print("[red]Results will be INVALID - warm page cache detected![/red]")
        console.print("[yellow]Run with: sudo -E ./benchmark-baseline.py[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[yellow]Warning: Cache drop failed: {e}[/yellow]")

    # 3. Clear kernel compilation caches (if persistent)
    cache_dirs = [
        Path.home() / ".cache/vllm/torch_compile_cache",
        Path.home() / ".triton/cache",
        Path.home() / ".nv/ComputeCache",
    ]
    cleared_any = False
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            console.print(f"[cyan]Clearing {cache_dir.name}...[/cyan]")
            try:
                subprocess.run(["rm", "-rf", str(cache_dir)], timeout=10, stderr=subprocess.DEVNULL)
                cleared_any = True
            except Exception:
                pass

    if cleared_any:
        console.print("[green]✓ Kernel compilation caches cleared[/green]")

    # 4. Wait for system stabilization
    console.print("[cyan]Waiting for system to stabilize...[/cyan]")
    time.sleep(3)

    # 5. Verify GPU is clean
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_mem = result.stdout.strip().split('\n')[0]
        console.print(f"[green]✓ GPU memory used: {gpu_mem} MiB[/green]")
    except Exception:
        pass

    console.print("[bold green]✓ System ready for TRUE cold start benchmark[/bold green]")


def start_vllm_container(config: dict[str, str]) -> tuple[float, float]:
    """
    Start vLLM container and measure container start time.

    Returns:
        Tuple of (t0, t1) where:
        - t0: Time before podman run
        - t1: Time after podman run returns
    """
    console.print("[cyan]Starting vLLM container...[/cyan]")

    # Build podman run command (same as create-checkpoint.py)
    cmd = [
        "podman", "run",
        "-d",  # Detached mode
        "--name", config["CONT_NAME"],
        "--device", "nvidia.com/gpu=all",
        "--security-opt", "label=disable",
        "-v", f"{config['NVIDIA_LIBS_PATH']}:/opt/nvidia:ro",
        "-v", f"{config['MODELS_CACHE']}:/root/.cache/huggingface:rw",
        "-p", f"{config['API_PORT']}:8000",
        "--shm-size", "8g",
        config["VLLM_IMAGE"],
        "--model", config["MODEL_ID"],
        "--gpu-memory-utilization", config["GPU_MEMORY_UTIL"],
        "--max-model-len", config["MAX_MODEL_LEN"],
        "--disable-log-requests",
    ]

    # T0: Start timer before podman run
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

        # T1: Container started (podman run returned)
        t1 = time.perf_counter()

        container_id = result.stdout.strip()
        console.print(f"[green]Container started: {container_id[:12]}[/green]")
        console.print(f"[dim]Container start time: {t1-t0:.3f}s[/dim]")

        return t0, t1

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error starting container:[/red]")
        console.print(f"[red]{e.stderr}[/red]")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        console.print("[red]Error: Container start timed out[/red]")
        sys.exit(1)


def wait_for_health_check(
    config: dict[str, str],
    t1: float,
) -> float:
    """
    Wait for vLLM API to become ready via health check.

    Args:
        config: Configuration dictionary
        t1: Container start time

    Returns:
        t2: Time when API became ready
    """
    base_url = f"http://localhost:{config['API_PORT']}"
    health_url = f"{base_url}/health"
    timeout = int(config["HEALTH_CHECK_TIMEOUT"])
    interval = float(config["HEALTH_CHECK_INTERVAL"])

    console.print(f"[cyan]Waiting for API to become ready at {health_url}[/cyan]")

    start_wait = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking health endpoint...", total=None)

        while (time.perf_counter() - start_wait) < timeout:
            try:
                with httpx.Client(timeout=2.0) as client:
                    response = client.get(health_url)

                    if response.status_code == 200:
                        # T2: API ready
                        t2 = time.perf_counter()

                        progress.update(task, description="[green]API ready![/green]")
                        console.print(f"[green]Health check passed![/green]")
                        console.print(f"[dim]Model load time: {t2-t1:.3f}s[/dim]")

                        return t2

            except (httpx.ConnectError, httpx.TimeoutException):
                pass  # Expected during startup
            except Exception as e:
                console.print(f"[yellow]Warning: Unexpected error during health check: {e}[/yellow]")

            time.sleep(interval)
            elapsed = time.perf_counter() - start_wait
            progress.update(
                task,
                description=f"Waiting for health... ({elapsed:.1f}s/{timeout}s)",
            )

    console.print(f"[red]Error: Health check timeout after {timeout}s[/red]")
    sys.exit(1)


def measure_inference_tfft(
    config: dict[str, str],
    t2: float,
) -> tuple[float, float, str]:
    """
    Send inference request and measure TFFT.

    Args:
        config: Configuration dictionary
        t2: API ready time

    Returns:
        Tuple of (t3, t4, full_response) where:
        - t3: Time of first token
        - t4: Time of full response
        - full_response: Complete response text
    """
    base_url = f"http://localhost:{config['API_PORT']}"
    completions_url = f"{base_url}/v1/completions"

    console.print("[cyan]Sending inference request...[/cyan]")

    # Prepare request payload
    payload = {
        "model": config["MODEL_ID"],
        "prompt": "Write a haiku about GPU computing:",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": True,  # Enable streaming to measure TFFT
    }

    t3 = None  # Time of first token
    t4 = None  # Time of full response
    full_response = ""

    try:
        with httpx.Client(timeout=30.0) as client:
            with client.stream("POST", completions_url, json=payload) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line.strip() or line.strip() == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        # Parse SSE data
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix

                            if "choices" in data and len(data["choices"]) > 0:
                                choice = data["choices"][0]

                                if "text" in choice:
                                    token = choice["text"]

                                    # T3: First token received
                                    if t3 is None:
                                        t3 = time.perf_counter()
                                        console.print(f"[green]First token received![/green]")
                                        console.print(f"[dim]Inference TFFT: {t3-t2:.3f}s[/dim]")

                                    full_response += token

                        except json.JSONDecodeError:
                            continue

        # T4: Full response received
        t4 = time.perf_counter()

        if t3 is None:
            console.print("[red]Error: No tokens received from inference[/red]")
            sys.exit(1)

        console.print(f"[green]Full response received![/green]")
        console.print(f"[dim]Total response time: {t4-t2:.3f}s[/dim]")

        # Show response
        console.print("\n[cyan]Response:[/cyan]")
        console.print(Panel(full_response.strip(), border_style="green"))

        return t3, t4, full_response

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: HTTP {e.response.status_code}[/red]")
        console.print(f"[red]{e.response.text}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error during inference: {e}[/red]")
        sys.exit(1)


def save_results(
    config: dict[str, str],
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> Path:
    """
    Save benchmark results to JSON file.

    Args:
        config: Configuration dictionary
        t0: Start time
        t1: Container started time
        t2: API ready time
        t3: First token time
        t4: Full response time

    Returns:
        Path to results file
    """
    results_dir = Path(config["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"baseline_{timestamp}.json"

    # Calculate metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "type": "baseline",
        "model": config["MODEL_ID"],
        "container_start_time": round(t1 - t0, 6),
        "model_load_time": round(t2 - t1, 6),
        "inference_tfft": round(t3 - t2, 6),
        "total_cold_start_tfft": round(t3 - t0, 6),
        "full_response_time": round(t4 - t0, 6),
    }

    # Save to file
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"\n[green]Results saved to: {results_file}[/green]")

    return results_file


def display_summary(
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    t4: float,
) -> None:
    """Display a summary table of all timing metrics."""

    table = Table(title="Baseline Benchmark Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Description", style="dim")

    table.add_row(
        "Container Start Time",
        f"{t1-t0:.6f}",
        "T1-T0: podman run execution",
    )
    table.add_row(
        "Model Load Time",
        f"{t2-t1:.6f}",
        "T2-T1: Model loading until health check passes",
    )
    table.add_row(
        "Inference TFFT",
        f"{t3-t2:.6f}",
        "T3-T2: Time to first token after API ready",
    )
    table.add_row(
        "Total Cold Start TFFT",
        f"{t3-t0:.6f}",
        "T3-T0: Total time from start to first token",
        style="bold green",
    )
    table.add_row(
        "Full Response Time",
        f"{t4-t0:.6f}",
        "T4-T0: Total time to complete response",
    )

    console.print()
    console.print(table)
    console.print()


def main() -> None:
    """Main benchmark execution flow."""

    console.print(Panel.fit(
        "[bold cyan]Baseline Cold Start TFFT Benchmark[/bold cyan]\n"
        "Measuring traditional vLLM cold start performance",
        border_style="cyan",
    ))
    console.print()

    # Step 1: Load environment
    console.print("[bold]Step 1: Loading environment configuration[/bold]")
    config = load_environment()
    console.print(f"[green]Configuration loaded[/green]")
    console.print(f"  Model: {config['MODEL_ID']}")
    console.print(f"  Container: {config['CONT_NAME']}")
    console.print(f"  Port: {config['API_PORT']}")
    console.print()

    # Step 2: Cleanup
    console.print("[bold]Step 2: Cleaning up existing resources[/bold]")
    cleanup_containers(config["CONT_NAME"])
    cleanup_for_true_cold_start()
    console.print("[green]Cleanup complete[/green]")
    console.print()

    try:
        # Step 3: Start container and measure T0, T1
        console.print("[bold]Step 3: Starting vLLM container[/bold]")
        t0, t1 = start_vllm_container(config)
        console.print()

        # Step 4: Wait for health check and measure T2
        console.print("[bold]Step 4: Waiting for API to become ready[/bold]")
        t2 = wait_for_health_check(config, t1)
        console.print()

        # Step 5: Send inference request and measure T3, T4
        console.print("[bold]Step 5: Measuring inference TFFT[/bold]")
        t3, t4, response = measure_inference_tfft(config, t2)
        console.print()

        # Step 6: Save results
        console.print("[bold]Step 6: Saving results[/bold]")
        results_file = save_results(config, t0, t1, t2, t3, t4)
        console.print()

        # Display summary
        display_summary(t0, t1, t2, t3, t4)

        console.print(Panel.fit(
            "[bold green]Benchmark completed successfully![/bold green]\n"
            f"Results: {results_file}",
            border_style="green",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        sys.exit(1)
    finally:
        # Step 7: Cleanup
        console.print("\n[bold]Step 7: Cleaning up container[/bold]")
        cleanup_containers(config["CONT_NAME"])
        console.print("[green]Cleanup complete[/green]")


if __name__ == "__main__":
    main()
