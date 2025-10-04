# Docker ROCm Notes

A comprehensive reference guide for running LLM models in Docker containers using AMD ROCm GPU acceleration.

## What This Repository Contains

- **[ROCM_DOCKER_GUIDE.md](ROCM_DOCKER_GUIDE.md)** - Complete guide covering:
  - ROCm Docker architecture and requirements
  - Official AMD images and documentation links
  - GPU device mapping methods (CDI, runtime, manual)
  - Framework implementations (llama.cpp, vLLM, Ollama)
  - Example Dockerfiles and Docker Compose configurations
  - Best practices for production deployments
  - Troubleshooting common issues

- **Docker Compose Examples** - Ready-to-use configurations for different use cases

## Quick Start

### Prerequisites

Host system requires only:
- AMD GPU kernel driver (`amdgpu-dkms`)
- Docker 25.0+
- User in `video` and `render` groups

No ROCm installation needed on host - all ROCm libraries run inside containers.

### Basic Usage

**llama.cpp (Recommended for most users):**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v ./models:/data \
  -p 8000:8000 \
  rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server \
  -m /data/model.gguf --host 0.0.0.0
```

**Ollama (Easiest setup):**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:rocm
```

**vLLM (Production/high performance):**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --security-opt seccomp=unconfined \
  --ipc=host --shm-size 16G \
  -v ./models:/models \
  -p 8000:8000 \
  rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909 \
  --model /models/llama-2-7b-hf --host 0.0.0.0
```

## Key Architecture Points

**Self-Contained Design:**
- Host: Only kernel driver (`amdgpu-dkms`)
- Container: All ROCm userspace libraries, runtimes, and frameworks

**Recommended Versions (2025):**
- ROCm 6.4.3: Most stable for production
- ROCm 7.0.1: Latest stable release
- Ubuntu 24.04 base images recommended

## Framework Comparison

| Framework | Best For | Model Format | Ease of Use | Performance |
|-----------|----------|--------------|-------------|-------------|
| **llama.cpp** | General use | GGUF | Medium | Good |
| **Ollama** | Development | GGUF (auto-convert) | Easy | Good |
| **vLLM** | Production API | HuggingFace | Medium | Excellent |

## Documentation Resources

### AMD Official Docs
- [ROCm Docker Guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
- [ROCm Containers Blog](https://rocm.blogs.amd.com/software-tools-optimization/rocm-containers/README.html)
- [Compatibility Matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)

### Framework-Specific
- [llama.cpp with ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/llama-cpp-install.html)
- [vLLM ROCm Guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html)
- [Ollama Docker Guide](https://docs.ollama.com/docker)

## GPU Device Access Methods

### Method 1: CDI (Recommended 2025)
```bash
docker run --device amd.com/gpu=all <image>
```
Requires: AMD Container Toolkit

### Method 2: AMD Container Runtime
```bash
docker run --runtime=amd -e AMD_VISIBLE_DEVICES=0 <image>
```

### Method 3: Manual Device Mounting
```bash
docker run --device=/dev/kfd --device=/dev/dri --group-add video <image>
```

## Common Issues

**GPU not detected:** Verify kernel driver loaded (`lsmod | grep amdgpu`) and device nodes exist (`ls -la /dev/kfd /dev/dri/`)

**Permission denied:** Add user to groups: `sudo usermod -a -G video,render $USER` (logout/login required)

**Out of memory:** Increase shared memory: `--shm-size 16G`

**Slow performance:** Verify GPU usage with `rocm-smi` and ensure GPU layers enabled (`-ngl 99`)

See [ROCM_DOCKER_GUIDE.md](ROCM_DOCKER_GUIDE.md) for detailed troubleshooting.

## License

See [LICENSE](LICENSE) file.

---

*Last Updated: October 2025*
