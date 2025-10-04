# Running LLM Models in Docker with AMD ROCm Backend

Complete guide for hosting local LLM models in Docker using AMD ROCm, with all runtime libraries self-contained within containers.

## Table of Contents

- [Overview](#overview)
- [Host Requirements](#host-requirements)
- [Official Documentation](#official-documentation)
- [Recommended Docker Images](#recommended-docker-images)
- [GPU Device Mapping](#gpu-device-mapping)
- [Implementation Options](#implementation-options)
- [Example Dockerfiles](#example-dockerfiles)
- [Best Practices 2025](#best-practices-2025)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Self-Contained Architecture

ROCm containers follow a **kernel/userspace split architecture**:

- **Host**: Only AMD GPU kernel driver (`amdgpu-dkms`)
- **Container**: All ROCm userspace libraries, runtimes, and frameworks

This ensures:
- No ROCm installation needed on host system
- Clean host environment
- Easy version management per container
- Reproducible deployments

### What Runs Inside Container

- HIP runtime
- ROCm libraries (rocBLAS, rocFFT, MIOpen, etc.)
- ROCm tools (`rocm-smi`, `rocminfo`)
- ML frameworks (PyTorch, TensorFlow, etc.)
- Inference engines (llama.cpp, vLLM, Ollama, etc.)

---

## Host Requirements

### Minimal Requirements (Nothing Extra!)

1. **AMD GPU Kernel Driver Only**
   ```bash
   # Install amdgpu-dkms (if not already installed)
   # Ubuntu/Debian:
   sudo apt-get install amdgpu-dkms

   # Fedora:
   sudo dnf install amdgpu-dkms
   ```

2. **Docker** (version 25.0+ recommended)
   ```bash
   docker --version
   ```

3. **User Group Permissions**
   ```bash
   # Add user to video and render groups
   sudo usermod -a -G video,render $USER

   # Log out and back in, then verify:
   groups | grep -E 'video|render'
   ```

4. **Verify Device Nodes Exist**
   ```bash
   ls -la /dev/kfd /dev/dri/
   ```

### What is NOT Needed on Host

❌ ROCm userspace libraries
❌ ROCm development tools
❌ HIP runtime
❌ PyTorch/TensorFlow
❌ Any ML frameworks

All these run **inside the container only**.

---

## Official Documentation

### AMD ROCm Core Documentation

- **ROCm Docker Guide**: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html
- **ROCm Containers Blog**: https://rocm.blogs.amd.com/software-tools-optimization/rocm-containers/README.html
- **ROCm Docker GitHub**: https://github.com/ROCm/ROCm-docker
- **Docker Hub**: https://hub.docker.com/u/rocm
- **Compatibility Matrix**: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

### Framework-Specific Documentation

#### llama.cpp with ROCm
- **AMD llama.cpp Guide**: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html
- **Docker Hub (AMD)**: https://hub.docker.com/r/rocm/llama.cpp
- **Compatibility**: https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/llama-cpp-compatibility.html
- **ggml-org Docker**: https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md

#### vLLM with ROCm
- **Docker Hub (AMD)**: https://hub.docker.com/r/rocm/vllm
- **vLLM ROCm Guide**: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html
- **vLLM Docs**: https://docs.vllm.ai/en/stable/deployment/docker.html

#### Ollama with ROCm
- **Official Docker Guide**: https://docs.ollama.com/docker
- **GitHub Docs**: https://github.com/ollama/ollama/blob/main/docs/docker.md

#### AMD Container Toolkit
- **Blog**: https://rocm.blogs.amd.com/software-tools-optimization/amd-container-toolkit/README.html
- **GitHub**: https://github.com/ROCm/container-toolkit
- **Documentation**: https://instinct.docs.amd.com/projects/container-toolkit/en/latest/

---

## Recommended Docker Images

### ROCm Base Images (2025)

#### Production Stable
```dockerfile
# Ubuntu 24.04 with ROCm 6.4 (Recommended)
FROM rocm/dev-ubuntu-24.04:6.4-complete

# Ubuntu 22.04 with ROCm 6.4 (Alternative)
FROM rocm/dev-ubuntu-22.04:6.4-complete
```

#### Latest Stable
```dockerfile
# ROCm 7.0.1 (Released Sept 2025)
FROM rocm/dev-ubuntu-24.04:7.0.1-complete
```

**Version Recommendations:**
- **ROCm 6.4.3**: Most stable for production workloads
- **ROCm 7.0.1**: Latest stable (verify application compatibility)
- **ROCm 7.0 RC**: Preview only, NOT for production

### Framework-Specific Images

#### llama.cpp (AMD Official)
```bash
# Server variant (recommended)
rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server

# Full variant (includes all tools)
rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_full

# Light variant (minimal size)
rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_light
```

#### vLLM (AMD Official)
```bash
# Latest stable vLLM with ROCm
rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
```

#### Ollama (Official)
```bash
# Ollama with ROCm support
ollama/ollama:rocm
```

#### PyTorch
```bash
# PyTorch with ROCm
rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.3.0
```

---

## GPU Device Mapping

Three methods for GPU access (ordered by recommendation):

### Method 1: Container Device Interface (CDI) - RECOMMENDED for 2025

```bash
# One-time setup: Generate CDI specification
sudo amd-ctk cdi generate --output=/etc/cdi/amd.json

# Run container with CDI
docker run --rm --device amd.com/gpu=all <image>

# Or specific GPU
docker run --rm --device amd.com/gpu=0 <image>
```

**Benefits:**
- Modern, standardized approach
- No manual device path management
- Works across container runtimes
- AMD's recommended method

**Requires:** AMD Container Toolkit on host

### Method 2: AMD Container Runtime

```bash
docker run --runtime=amd \
  -e AMD_VISIBLE_DEVICES=0,1 \
  <image>
```

**Requires:** AMD Container Toolkit with runtime hook

### Method 3: Manual Device Mounting - TRADITIONAL

```bash
docker run \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  <image>
```

**Device Details:**
- `/dev/kfd`: Kernel Fusion Driver (compute interface, shared by all GPUs)
- `/dev/dri`: Direct Rendering Infrastructure
  - `/dev/dri/renderD128`: GPU 0
  - `/dev/dri/renderD129`: GPU 1
  - `/dev/dri/renderD130`: GPU 2 (etc.)

**Common Options Explained:**

| Flag | Purpose | Required? |
|------|---------|-----------|
| `--device=/dev/kfd` | GPU compute access | Yes |
| `--device=/dev/dri` | GPU rendering/display | Yes |
| `--security-opt seccomp=unconfined` | Enable memory mapping | Recommended for HPC |
| `--group-add video` | Video group access | Yes |
| `--ipc=host` | Shared memory for multi-GPU | For multi-GPU/multi-process |
| `--shm-size 16G` | Increase shared memory | For large models |
| `--cap-add=SYS_PTRACE` | Debugging capability | Optional (debug only) |
| `--privileged` | Full system access | ❌ NOT recommended |

---

## Implementation Options

### Option 1: llama.cpp (Recommended for Most Users)

**Pros:**
- Official AMD-supported images
- Lightweight and fast
- Excellent GGUF model support
- Simple HTTP API
- Pre-built and validated

**Use Case:** General-purpose local inference, chatbots, embeddings

**Quick Start:**
```bash
# Pull image
docker pull rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server

# Run server
docker run -d \
  --name llama-server \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -v $(pwd)/models:/data \
  -p 8000:8000 \
  rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server \
  -m /data/model.gguf --host 0.0.0.0 --port 8000

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Option 2: Ollama (Easiest Setup)

**Pros:**
- Simplest to use
- Built-in model library
- Automatic model downloads
- CLI and API
- Great for beginners

**Use Case:** Quick prototyping, development, personal use

**Quick Start:**
```bash
# Run Ollama
docker run -d \
  --name ollama \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:rocm

# Pull and run a model
docker exec -it ollama ollama run llama3.2

# Or via API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?"
}'
```

### Option 3: vLLM (High Performance)

**Pros:**
- Highest throughput
- OpenAI-compatible API
- Advanced batching
- Production-grade
- Optimized for Instinct GPUs

**Use Case:** Production deployments, API services, high-traffic apps

**Quick Start:**
```bash
# Pull image
docker pull rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909

# Run server
docker run -d \
  --name vllm-server \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 16G \
  -v $(pwd)/models:/models \
  -p 8000:8000 \
  rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909 \
  --model /models/llama-2-7b-hf --host 0.0.0.0

# Test (OpenAI-compatible)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/llama-2-7b-hf",
    "prompt": "Once upon a time",
    "max_tokens": 50
  }'
```

### Comparison Table

| Feature | llama.cpp | Ollama | vLLM |
|---------|-----------|--------|------|
| **Ease of Use** | Medium | Easy | Medium |
| **Performance** | Good | Good | Excellent |
| **Memory Efficient** | Yes | Yes | Moderate |
| **Model Format** | GGUF | GGUF (auto-convert) | HuggingFace |
| **API** | OpenAI-like | Custom + OpenAI | OpenAI-compatible |
| **Production Ready** | Yes | Development | Yes |
| **AMD Support** | Official | Community | Official |
| **Best For** | General use | Development | Production API |

---

## Example Dockerfiles

### 1. llama.cpp Server (Build from Source)

```dockerfile
FROM rocm/dev-ubuntu-24.04:6.4-complete

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
WORKDIR /workspace
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1100,gfx1101,gfx1102 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON && \
    cmake --build build --config Release -j$(nproc)

# Create volume for models
VOLUME ["/data"]

# Expose API port
EXPOSE 8000

WORKDIR /workspace/llama.cpp
ENTRYPOINT ["./build/bin/llama-server"]
CMD ["--help"]
```

**GPU Architecture Targets (AMDGPU_TARGETS):**

| GPU Series | Target | Examples |
|------------|--------|----------|
| **Instinct MI50** | `gfx906` | MI50 |
| **Instinct MI100** | `gfx908` | MI100 |
| **Instinct MI200** | `gfx90a` | MI250X, MI250, MI210 |
| **Instinct MI300** | `gfx942` | MI300X, MI300A |
| **Radeon RX 6000** | `gfx1030` | RX 6900 XT, 6800 XT, 6700 XT |
| **Radeon RX 7000** | `gfx1100` | RX 7900 XTX, 7900 XT |
| | `gfx1101` | RX 7800 XT, 7700 XT |
| | `gfx1102` | RX 7600 XT, 7600 |
| **Radeon RX 9000** | `gfx1201` | RX 9070 XT, 9070 |

**Build Example:**
```bash
# Build for RX 7900 XT
docker build -t llama-rocm:rx7900 .

# Run server
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v ./models:/data \
  -p 8000:8000 \
  llama-rocm:rx7900 \
  -m /data/llama-2-7b.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8000 -ngl 99
```

### 2. vLLM Custom Dockerfile

```dockerfile
FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Install vLLM
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir vllm

# Set environment variables for GPU architecture
ENV PYTORCH_ROCM_ARCH="gfx90a;gfx942"
ENV HIP_VISIBLE_DEVICES=0

# Create model directory
VOLUME ["/models"]

# Expose API port
EXPOSE 8000

WORKDIR /app

# vLLM OpenAI-compatible server
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--help"]
```

**Usage:**
```bash
# Build
docker build -t vllm-rocm:custom .

# Run
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v ./models:/models \
  -p 8000:8000 \
  vllm-rocm:custom \
  --model meta-llama/Llama-2-7b-hf \
  --host 0.0.0.0 --port 8000
```

### 3. Multi-Stage Build (Optimized Size)

```dockerfile
# Build stage
FROM rocm/dev-ubuntu-24.04:6.4-complete AS builder

RUN apt-get update && apt-get install -y \
    cmake git libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -S . -B build \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS=gfx1100,gfx1101 \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Runtime stage
FROM rocm/rocm-terminal:6.4

# Copy only runtime libraries and binary
COPY --from=builder /opt/rocm /opt/rocm
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    libcurl4 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH}"

VOLUME ["/data"]
EXPOSE 8000

ENTRYPOINT ["llama-server"]
CMD ["--help"]
```

### 4. Docker Compose Setup

```yaml
version: '3.8'

services:
  llama-server:
    image: rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server
    container_name: llama-server
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp=unconfined
    group_add:
      - video
    ipc: host
    shm_size: 16g
    volumes:
      - ./models:/data:ro
    ports:
      - "8000:8000"
    command:
      - "-m"
      - "/data/llama-2-7b.Q4_K_M.gguf"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "-ngl"
      - "99"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  vllm-server:
    image: rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
    container_name: vllm-server
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp=unconfined
    group_add:
      - video
    ipc: host
    shm_size: 16g
    volumes:
      - ./models:/models:ro
    ports:
      - "8001:8000"
    command:
      - "--model"
      - "/models/llama-2-7b-hf"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
    restart: unless-stopped
    environment:
      - HIP_VISIBLE_DEVICES=0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:rocm
    container_name: ollama
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    volumes:
      - ollama-data:/root/.ollama
    ports:
      - "11434:11434"
    restart: unless-stopped

volumes:
  ollama-data:
```

**Usage:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 5. Development Container (Interactive)

```dockerfile
FROM rocm/dev-ubuntu-24.04:6.4-complete

# Install development tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    vim \
    htop \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML packages
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/rocm6.0

RUN pip3 install --no-cache-dir \
    transformers \
    accelerate \
    huggingface-hub \
    datasets \
    jupyter \
    ipython

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Set environment variables
ENV HIP_VISIBLE_DEVICES=0
ENV ROCR_VISIBLE_DEVICES=0
ENV PYTORCH_ROCM_ARCH="gfx1100"

CMD ["/bin/bash"]
```

**Usage:**
```bash
# Build
docker build -t rocm-dev:6.4 .

# Run interactive
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v $(pwd):/workspace \
  rocm-dev:6.4

# Inside container - test GPU
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Best Practices 2025

### Architecture & Design

1. **Kernel/Userspace Split**
   - Host: Only `amdgpu-dkms` kernel driver
   - Container: All ROCm userspace libraries and tools

2. **Use Official AMD Images**
   - Base images: `rocm/dev-ubuntu-24.04:6.4-complete`
   - Framework images: `rocm/llama.cpp`, `rocm/vllm`, etc.

3. **Version Pinning**
   - Always use specific tags, never `:latest`
   - Example: `rocm/dev-ubuntu-24.04:6.4.3-complete`

4. **Multi-Stage Builds**
   - Separate build and runtime stages
   - Minimize final image size
   - Include only necessary runtime dependencies

### Security

1. **Avoid `--privileged`**
   - Use specific `--device` flags instead
   - Principle of least privilege

2. **Use CDI (Modern Method)**
   - Container Device Interface for GPU access
   - Standardized and future-proof

3. **Minimal Security Options**
   - Only use `seccomp=unconfined` when necessary
   - Required for HPC/high-performance workloads

4. **Non-Root User**
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

### Performance

1. **IPC Mode**
   - Use `--ipc=host` for multi-GPU or multi-process workloads
   - Enables shared memory communication

2. **Shared Memory**
   - Set `--shm-size 16G` or higher for large models
   - Default 64MB is insufficient

3. **GPU Architecture Targeting**
   - Build for specific `AMDGPU_TARGETS`
   - Example: `gfx1100` for RX 7900 XT
   - Optimizes kernel performance

4. **Network Mode**
   - Consider `--network=host` for low-latency inference
   - Bypasses Docker network overhead

### Stability

1. **ROCm Version Selection**
   - **ROCm 6.4.3**: Most stable for production (early 2025)
   - **ROCm 7.0.1**: Latest stable (Sept 2025) - verify compatibility
   - Avoid release candidates in production

2. **Version Compatibility**
   - Match host kernel driver with container ROCm runtime
   - Check compatibility matrix: https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

3. **Health Checks**
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
     CMD curl -f http://localhost:8000/health || exit 1
   ```

4. **Restart Policies**
   ```bash
   docker run --restart unless-stopped ...
   ```

### Resource Management

1. **Memory Limits**
   ```bash
   docker run --memory=32g --memory-swap=32g ...
   ```

2. **CPU Pinning**
   ```bash
   docker run --cpuset-cpus=0-7 ...
   ```

3. **GPU Selection (Multi-GPU)**
   ```bash
   # Method 1: Environment variable
   docker run -e HIP_VISIBLE_DEVICES=0,1 ...

   # Method 2: Specific devices
   docker run --device=/dev/dri/renderD128 ...
   ```

### Monitoring & Debugging

1. **Include Monitoring Tools**
   ```dockerfile
   # ROCm tools already in rocm/dev images
   # rocm-smi, rocminfo, rocprof
   ```

2. **View GPU Usage**
   ```bash
   # Inside container
   docker exec -it <container> rocm-smi

   # Watch GPU usage
   docker exec -it <container> watch -n 1 rocm-smi
   ```

3. **Enable Logging**
   ```bash
   docker run -e ROCM_LOGGING=1 ...
   ```

4. **Debug Mode**
   ```bash
   docker run -e HSA_ENABLE_SDMA=0 -e AMD_LOG_LEVEL=3 ...
   ```

### GPU Support Notes (2025)

| GPU Series | Support Level | Notes |
|------------|---------------|-------|
| **Instinct MI100/200/300** | Full Official | Production-ready, best performance |
| **Radeon RX 7900 XTX/XT** | Official (6.4.4+) | PyTorch on Linux, some limitations |
| **Radeon RX 7800/7700 XT** | Official (6.4.4+) | Consumer GPUs, good for development |
| **Radeon RX 7600** | Official (6.4.4+) | Entry-level, limited VRAM |
| **Radeon RX 6900/6800 XT** | Limited | Runtime only, no SDK support |
| **Radeon RX 6700 XT** | Limited | Runtime only |
| **Radeon VII** | Supported | Good performance on Linux |
| **Older GPUs** | Check compatibility matrix | Varies by model |

---

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

**Symptom:**
```
No ROCm devices found
```

**Solutions:**

```bash
# Check kernel driver loaded
lsmod | grep amdgpu

# Check device nodes exist
ls -la /dev/kfd /dev/dri/

# Check user permissions
groups | grep -E 'video|render'

# Inside container, check GPU visibility
docker exec -it <container> rocm-smi

# Verify device mapping
docker inspect <container> | grep -A 10 Devices
```

#### 2. Permission Denied

**Symptom:**
```
Error: Permission denied accessing /dev/kfd
```

**Solutions:**

```bash
# Add user to groups (then logout/login)
sudo usermod -a -G video,render $USER

# Verify group membership
id

# Check device permissions
ls -la /dev/kfd /dev/dri/

# Run with explicit group
docker run --group-add $(getent group video | cut -d: -f3) ...
```

#### 3. Out of Memory

**Symptom:**
```
RuntimeError: HIP out of memory
```

**Solutions:**

```bash
# Increase shared memory
docker run --shm-size 16G ...

# Check GPU memory
docker exec -it <container> rocm-smi

# Use smaller model or quantization
# Q4_K_M instead of Q8_0

# Limit context size
llama-server -m model.gguf -c 2048  # instead of 4096
```

#### 4. Slow Performance

**Symptom:**
- Low tokens/second
- High latency

**Solutions:**

```bash
# Verify GPU is being used
docker exec -it <container> rocm-smi

# Check GPU architecture match
# Build with correct -DAMDGPU_TARGETS

# Enable GPU layers
llama-server -m model.gguf -ngl 99

# Use IPC host mode
docker run --ipc=host ...

# Verify no CPU fallback
docker logs <container> | grep -i cpu
```

#### 5. Version Mismatch

**Symptom:**
```
HSA Error: Incompatible kernel and userspace
```

**Solutions:**

```bash
# Check host kernel driver version
modinfo amdgpu | grep version

# Check container ROCm version
docker run --rm <image> cat /opt/rocm/.info/version

# Use matching versions
# Host: amdgpu 6.4.x -> Container: rocm 6.4.x

# Consult compatibility matrix
# https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html
```

#### 6. Model Loading Fails

**Symptom:**
```
Error loading model: file not found
```

**Solutions:**

```bash
# Verify volume mount
docker inspect <container> | grep -A 5 Mounts

# Check file permissions
ls -la models/

# Ensure model path is correct
docker exec -it <container> ls -la /data/

# Test with absolute path
docker run -v $(pwd)/models:/data ...
```

#### 7. Network/API Issues

**Symptom:**
- Cannot connect to API
- Port not accessible

**Solutions:**

```bash
# Check port mapping
docker ps

# Verify service is listening
docker exec -it <container> netstat -tlnp

# Check firewall
sudo ufw status

# Test from container
docker exec -it <container> curl http://localhost:8000/health

# Test from host
curl http://localhost:8000/health
```

### Diagnostic Commands

```bash
# Check ROCm installation in container
docker run --rm <image> rocminfo

# List available GPUs
docker run --rm --device=/dev/kfd --device=/dev/dri <image> rocm-smi

# Verify HIP installation
docker run --rm <image> hipconfig --version

# Check PyTorch GPU support
docker run --rm <image> python3 -c "import torch; print(torch.cuda.is_available())"

# Check ROCm version
docker run --rm <image> cat /opt/rocm/.info/version

# List GPU compute capabilities
docker run --rm --device=/dev/kfd --device=/dev/dri <image> rocminfo | grep -i gfx
```

### Performance Benchmarking

```bash
# llama.cpp benchmark
docker exec -it llama-server \
  /workspace/llama.cpp/build/bin/llama-bench \
  -m /data/model.gguf -ngl 99

# GPU memory bandwidth test
docker exec -it <container> rocm-bandwidth-test

# GPU compute benchmark
docker exec -it <container> /opt/rocm/bin/rocprof --stats python3 test.py
```

### Getting Help

1. **Check Logs**
   ```bash
   docker logs <container>
   docker logs -f <container>  # follow mode
   ```

2. **AMD ROCm GitHub Issues**
   - https://github.com/ROCm/ROCm/issues

3. **Framework-Specific Issues**
   - llama.cpp: https://github.com/ggml-org/llama.cpp/issues
   - vLLM: https://github.com/vllm-project/vllm/issues
   - Ollama: https://github.com/ollama/ollama/issues

4. **AMD Community**
   - https://community.amd.com/

---

## Quick Reference

### Minimal Docker Run Command

```bash
docker run -d \
  --name my-llm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -v ./models:/data \
  -p 8000:8000 \
  rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server \
  -m /data/model.gguf --host 0.0.0.0
```

### Production Docker Run Command

```bash
docker run -d \
  --name my-llm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  --shm-size 16G \
  --memory 32G \
  --restart unless-stopped \
  -v ./models:/data:ro \
  -p 8000:8000 \
  -e HIP_VISIBLE_DEVICES=0 \
  rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server \
  -m /data/model.gguf --host 0.0.0.0 --port 8000 -ngl 99
```

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `HIP_VISIBLE_DEVICES` | Select GPU(s) | `0`, `0,1`, `all` |
| `ROCR_VISIBLE_DEVICES` | Alternative GPU selector | `0` |
| `HSA_OVERRIDE_GFX_VERSION` | Override GPU arch (for unsupported GPUs) | `10.3.0` for RX 6000 |
| `ROCM_PATH` | ROCm installation path | `/opt/rocm` |
| `AMD_LOG_LEVEL` | ROCm logging level | `0-4` (0=off, 4=verbose) |
| `GPU_MAX_HW_QUEUES` | Max hardware queues | `8` |

### Testing GPU in Container

```bash
# Run test container
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  rocm/dev-ubuntu-24.04:6.4-complete \
  bash

# Inside container:
rocm-smi                    # GPU status
rocminfo                    # Detailed GPU info
/opt/rocm/bin/hipconfig     # HIP configuration
```

---

## Summary

### Key Takeaways

✅ **Host needs ONLY kernel driver** (`amdgpu-dkms`) - all ROCm userspace in container
✅ **ROCm 6.4.3 recommended** for production stability (2025)
✅ **Use official AMD images**: `rocm/llama.cpp`, `rocm/vllm`, `rocm/dev-ubuntu-24.04`
✅ **Device access**: `/dev/kfd` + `/dev/dri` required (CDI preferred for 2025)
✅ **llama.cpp**: AMD provides official images - easiest to use
✅ **Version compatibility**: Match host kernel driver with container ROCm runtime
✅ **Security**: `--security-opt seccomp=unconfined` + `--group-add video`
✅ **Resources**: `--ipc=host` + `--shm-size 16G` for large models

### Recommended Setup (2025)

**For most users:**
- Framework: **llama.cpp** (official AMD image)
- Base: `rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server`
- Models: GGUF format (Q4_K_M for balance)
- GPU access: Manual device mounting (simplest) or CDI (modern)

**For production:**
- Framework: **vLLM** (highest performance)
- Base: `rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909`
- Models: HuggingFace format
- GPU access: CDI with AMD Container Toolkit
- Orchestration: Docker Compose or Kubernetes

**For development:**
- Framework: **Ollama** (easiest)
- Base: `ollama/ollama:rocm`
- Models: Auto-downloaded from library
- GPU access: Manual device mounting

---

## Additional Resources

- **ROCm Installation Guide**: https://rocm.docs.amd.com/projects/install-on-linux/
- **AMD Instinct GPU Docs**: https://www.amd.com/en/products/accelerators/instinct.html
- **ROCm Performance Tuning**: https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

*Last Updated: October 2025*
*Based on ROCm 6.4.3/7.0.1 and official AMD documentation*
