# GPU-Load Checkpoint/Restore Documentation Index

Complete reference guide for understanding and using the CRIU checkpoint/restore system for vLLM.

## Documents

### 1. **CHECKPOINT_RESTORE_SUMMARY.md** (START HERE)
Quick overview of what the system does and how it works.
- **Best for**: Understanding the big picture
- **Read time**: 5 minutes
- **Covers**: Problem statement, architecture, key commands, performance

### 2. **CHECKPOINT_RESTORE_ANALYSIS.md** (DETAILED)
Comprehensive technical analysis of the entire workflow.
- **Best for**: Deep understanding of implementation details
- **Read time**: 20-30 minutes
- **Covers**: Step-by-step workflows, all configuration, benchmark details, optimization opportunities

### 3. **COMMANDS_REFERENCE.md** (PRACTICAL)
Complete reference of all actual commands used.
- **Best for**: Running operations, copy-paste commands
- **Read time**: Reference (look up as needed)
- **Covers**: All checkpoint, restore, testing, and debugging commands

### 4. **README.md** (PROJECT OVERVIEW)
Original project README with architecture diagrams.
- **Best for**: Project context and design rationale
- **Covers**: Problem statement, architecture, requirements, quick start

### 5. **README-checkpoint.md** (CHECKPOINT SCRIPT)
Detailed documentation of the checkpoint creation script.
- **Best for**: Understanding `create-checkpoint.py`
- **Covers**: Features, workflow, configuration, usage examples

## Reading Paths

### For Quick Understanding
1. CHECKPOINT_RESTORE_SUMMARY.md (5 min)
2. COMMANDS_REFERENCE.md (10 min) - scan the key commands section

### For Implementation
1. CHECKPOINT_RESTORE_SUMMARY.md (5 min)
2. CHECKPOINT_RESTORE_ANALYSIS.md - focus on sections 1 & 2 (10 min)
3. COMMANDS_REFERENCE.md (reference as needed)

### For Troubleshooting
1. COMMANDS_REFERENCE.md - jump to Common Troubleshooting section
2. CHECKPOINT_RESTORE_ANALYSIS.md - section 9 (Key Insights)
3. README.md - Architecture section

### For Deep Dive
1. CHECKPOINT_RESTORE_SUMMARY.md (5 min)
2. README.md (15 min)
3. CHECKPOINT_RESTORE_ANALYSIS.md (30 min) - complete
4. COMMANDS_REFERENCE.md (reference)

## Key Sections by Topic

### Understanding Checkpoints
- CHECKPOINT_RESTORE_SUMMARY.md - "What Gets Checkpointed"
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 1.5 "Checkpoint Creation Command"
- README-checkpoint.md - "Script Workflow"

### Creating Checkpoints
- COMMANDS_REFERENCE.md - "Checkpoint Creation" section
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 1 complete
- README-checkpoint.md - "Usage" section

### Restoring from Checkpoints
- COMMANDS_REFERENCE.md - "Restore Operations" section
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 2 complete
- CHECKPOINT_RESTORE_SUMMARY.md - "Key Commands" subsection

### Performance Benchmarking
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 7
- COMMANDS_REFERENCE.md - "Performance Measurement" section
- CHECKPOINT_RESTORE_SUMMARY.md - "Performance Numbers"

### Configuration
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 3
- CHECKPOINT_RESTORE_SUMMARY.md - "Critical Configuration" table
- COMMANDS_REFERENCE.md - "Environment Setup" section

### Troubleshooting
- COMMANDS_REFERENCE.md - "Common Troubleshooting" section
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 9.2 "Performance Bottlenecks"

### Storage & Files
- CHECKPOINT_RESTORE_ANALYSIS.md - Section 5
- COMMANDS_REFERENCE.md - "Checkpoint File Operations" section

## Quick Reference Tables

### File Purposes
| File | Purpose |
|------|---------|
| `.env` | Configuration (model, container name, ports) |
| `create-checkpoint.py` | Create golden checkpoint |
| `run-benchmarks.py` | Run full benchmark suite |
| `benchmark-criu.py` | Benchmark CRIU restore |
| `benchmark-baseline.py` | Benchmark cold start |
| `test-restore-speed.sh` | Test restore performance |
| `test-ignore-flags.sh` | Test network compatibility |
| `test-import-restore.sh` | Test tar import/restore |
| `checkpoint-analysis.py` | Analyze checkpoint contents |
| `analyze-results.py` | Analyze benchmark results |

### Command Categories
| Category | Where | Reference |
|----------|-------|-----------|
| Checkpoint creation | COMMANDS_REFERENCE.md | "Checkpoint Creation" |
| Restore operations | COMMANDS_REFERENCE.md | "Restore Operations" |
| Container management | COMMANDS_REFERENCE.md | "Container Management" |
| File operations | COMMANDS_REFERENCE.md | "Checkpoint File Operations" |
| Performance measurement | COMMANDS_REFERENCE.md | "Performance Measurement" |
| Troubleshooting | COMMANDS_REFERENCE.md | "Common Troubleshooting" |

## Key Concepts

### Checkpoint
A saved snapshot of a container's complete state including:
- Process memory (CPU RAM)
- GPU VRAM (model weights, KV cache)
- File descriptors and network sockets
- CUDA context state
- Python interpreter state

**Location**: `/var/lib/containers/storage/overlay-containers/{ID}/userdata/checkpoint/`
**Size**: Typically 2-5 GB

### Restore
Reconstruction of a container from a checkpoint snapshot.

**Commands**:
- `podman container restore {container}` - restore (container stops after)
- `podman container restore --keep {container}` - restore and keep running

**Performance**: 5-10 seconds (vs 20-40+ seconds cold start)

### vLLM
Large Language Model inference server.

**Configuration**: Qwen/Qwen2-1.5B-Instruct model (1.5B parameters)
**Features**: 
- OpenAI-compatible API
- CUDA-accelerated inference
- KV cache management
- Multi-token generation

### CRIU
Checkpoint/Restore In Userspace - Linux tool for checkpoint/restore.

**Used for**:
- Capturing process state
- Restoring process state
- Container checkpoint/restore

### TFFT
Time to First Token - latency metric for LLM inference.

**Measured from**: When inference request is sent to when first token is received
**Baseline**: 1-2 seconds (with model pre-loaded)
**CRIU**: 0.5-1 second (model in cache)

## Environment Variables

### Configuration File (.env)
```bash
MODEL_ID                  # HuggingFace model to use
CONT_NAME                 # Container name
API_PORT                  # API listen port
CHECKPOINT_DIR            # Checkpoint storage directory
HEALTH_CHECK_TIMEOUT      # How long to wait for health check
```

### Container Environment
```bash
LD_LIBRARY_PATH           # CUDA runtime library path
ASYNCIO_DEFAULT_BACKEND   # Set to "select" for CRIU compatibility
PYTHON_ASYNCIO_NO_IO_URING  # Set to 1 to disable io_uring
```

## Performance Expectations

### Baseline (Cold Start)
- Container startup: 2-5 seconds
- Model loading: 15-30 seconds
- Health check: 1-2 seconds
- **Total**: 20-40+ seconds

### CRIU Restore
- Checkpoint restore: 3-7 seconds
- Health check: 1-2 seconds
- **Total**: 5-10 seconds

### Speedup: **4-8x faster**

## Common Tasks

### Create a checkpoint
1. Read: CHECKPOINT_RESTORE_SUMMARY.md
2. Command: `./create-checkpoint.py`
3. Reference: COMMANDS_REFERENCE.md - "Checkpoint Creation"

### Run benchmarks
1. Read: CHECKPOINT_RESTORE_SUMMARY.md - "Workflow Example"
2. Command: `./run-benchmarks.py`
3. Reference: COMMANDS_REFERENCE.md - "Benchmark Scripts"

### Measure restore speed
1. Read: CHECKPOINT_RESTORE_ANALYSIS.md - Section 2.4
2. Command: `./test-restore-speed.sh`
3. Reference: COMMANDS_REFERENCE.md - "Performance Measurement"

### Debug issues
1. Read: COMMANDS_REFERENCE.md - "Common Troubleshooting"
2. Check: Container logs with `podman logs {container}`
3. Verify: Requirements with commands in "Environment Setup"

### Understand the architecture
1. Read: CHECKPOINT_RESTORE_SUMMARY.md - "Architecture"
2. Read: README.md - full document
3. Deep dive: CHECKPOINT_RESTORE_ANALYSIS.md - all sections

## Important Files & Locations

| Location | Purpose |
|----------|---------|
| `/root/gpu-load/.env` | Configuration file |
| `/root/gpu-load/*.py` | Python scripts |
| `/root/gpu-load/*.sh` | Bash test scripts |
| `/root/gpu-load/results/` | Benchmark results |
| `/var/lib/containers/storage/overlay-containers/*/userdata/checkpoint/` | Checkpoint files |
| `/mnt/checkpoint-ram/` | Optional fast checkpoint storage |
| `/models/` | Model cache (HuggingFace) |

## Workflow

```
┌─ START HERE ──────────────────────────┐
│                                       │
│  Read CHECKPOINT_RESTORE_SUMMARY.md   │
│                                       │
└────────────┬────────────────────────┘
             │
             ├─ Want to RUN?
             │  └─ COMMANDS_REFERENCE.md
             │
             ├─ Want to UNDERSTAND?
             │  └─ CHECKPOINT_RESTORE_ANALYSIS.md
             │
             ├─ Need DETAILS?
             │  ├─ README.md
             │  ├─ README-checkpoint.md
             │  └─ COMMANDS_REFERENCE.md
             │
             └─ Need HELP?
                └─ COMMANDS_REFERENCE.md - Troubleshooting
```

## Document Statistics

| Document | Lines | Topics | Read Time |
|----------|-------|--------|-----------|
| CHECKPOINT_RESTORE_SUMMARY.md | ~250 | 15 | 5 min |
| CHECKPOINT_RESTORE_ANALYSIS.md | ~700 | 50+ | 20-30 min |
| COMMANDS_REFERENCE.md | ~550 | 30+ | Reference |
| README.md | ~200 | 10 | 10 min |
| README-checkpoint.md | ~200 | 10 | 10 min |

---

**Total Documentation**: ~1,900 lines of comprehensive guides

Start with **CHECKPOINT_RESTORE_SUMMARY.md** for a quick overview, then choose your path based on your needs!
