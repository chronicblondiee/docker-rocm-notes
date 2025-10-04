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

### Using Docker Compose (Recommended)

All Docker Compose files support environment variable configuration. See [.env.example](.env.example) for all available options.

**Quick Start (using defaults):**

**Ollama (Easiest for development):**
```bash
docker-compose -f docker-compose-ollama.yml up -d
docker exec -it ollama-dev ollama pull llama3.2
docker exec -it ollama-dev ollama run llama3.2
```

**llama.cpp (General-purpose):**
```bash
mkdir -p models  # Place your .gguf models here
docker-compose -f docker-compose-llamacpp.yml up -d
```

**vLLM (Production/high performance):**
```bash
mkdir -p models hf-cache  # Place your HuggingFace models here
docker-compose -f docker-compose-vllm.yml up -d
```

**Custom Configuration:**

**Method 1: Using a .env file (Recommended)**

1. Create your `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your settings:
   ```bash
   # Example for llama.cpp with local models
   MODELS_DIR=./models
   MODEL_PATH=/data/granite-4.0-h-tiny-Q4_K_M.gguf
   LLAMA_PORT=8000
   GPU_ID=0
   ```

3. Start the service (automatically reads `.env`):
   ```bash
   docker-compose -f docker-compose-llamacpp.yml up -d
   ```

**Method 2: Using environment variables**

```bash
export MODELS_DIR="$HOME/.lmstudio/models"
export MODEL_PATH="/data/model-dir/model.gguf"
export GPU_ID=0
docker-compose -f docker-compose-llamacpp.yml up -d
```

**Method 3: Inline variables**

```bash
MODELS_DIR=./models MODEL_PATH=/data/my-model.gguf docker-compose -f docker-compose-llamacpp.yml up -d
```

**Key environment variables:**
- `MODELS_DIR` - Host directory containing models (default: `./models`)
- `MODEL_PATH` - Path to model file inside container (default: `/data/model.gguf`)
- `GPU_ID` - GPU device ID: `0`, `1`, or `"0,1"` for multi-GPU (default: `0`)
- `ROCM_ARCH` - GPU architecture, e.g., `gfx1100` for RX 7900 XTX (default: `gfx1100`)
- `HF_TOKEN` - HuggingFace token for gated models (vLLM only)

See [.env.example](.env.example) for all available configuration options and examples.

### Using Docker CLI

**llama.cpp:**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v ./models:/data \
  -p 8000:8000 \
  rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server \
  -m /data/model.gguf --host 0.0.0.0
```

**Ollama:**
```bash
docker run -d \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:rocm
```

**vLLM:**
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

## OpenAI API Compatibility

All backends provide **OpenAI-compatible APIs**, allowing you to use the official OpenAI SDK or any compatible client library with locally hosted models.

### Supported Features

| Feature | llama.cpp | Ollama | vLLM | Notes |
|---------|-----------|--------|------|-------|
| `/v1/chat/completions` | ✅ | ✅ | ✅ | Full support with streaming |
| `/v1/completions` | ✅ | ✅ | ✅ | Legacy text completion |
| `/v1/embeddings` | ✅ | ✅ | ✅ | Text embeddings |
| **Tool/Function Calling** | ✅ * | ✅ | ✅ | *Requires `--jinja` flag |
| **Streaming** | ✅ | ✅ | ✅ | Server-sent events |
| **JSON Mode** | ✅ | ✅ | ✅ | Structured output |

\* llama.cpp requires `--jinja` flag - **already configured** in [docker-compose-llamacpp.yml](docker-compose-llamacpp.yml)

### Quick Start with OpenAI SDK

**Python Example:**
```python
from openai import OpenAI

# llama.cpp
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Ollama
# client = OpenAI(
#     base_url="http://localhost:11434/v1",
#     api_key="not-needed"
# )

# Chat
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Tool calling (requires compatible model like Llama 3.1+)
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools
)
```

### Tool/Function Calling Setup

**llama.cpp:**
- ✅ `--jinja` flag enabled by default in our docker-compose
- ✅ Use compatible model (Llama 3.1+, Mistral Nemo, Qwen 2.5, Hermes)
- ✅ Q4_K_M or higher quantization recommended

**Ollama:**
- ✅ No setup required - works automatically
- ✅ Pull a compatible model: `docker exec -it ollama-dev ollama pull llama3.1`

**Recommended Models for Tool Calling:**
- **Llama 3.1 / 3.3** (8B, 70B) - Best overall choice
- **Mistral Nemo** (12B) - Excellent performance
- **Qwen 2.5** (7B, 14B, 32B) - Great multilingual support
- **Hermes 2/3** - Function calling focused

**Not Compatible:**
- ❌ Granite models (no tool support)
- ❌ Llama 3.2 1B/3B (no tool support)

📖 **For complete documentation, examples, and troubleshooting, see [OPENAI_API_GUIDE.md](OPENAI_API_GUIDE.md)**

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

## Testing

See [llama-tests/](llama-tests/) directory for test scripts:

- **test-llama.sh** - Basic functionality and API validation
- **load-test-llama.sh** - Load testing and performance benchmarks
- **concurrency-test-llama.sh** - Parallel vs sequential execution analysis

Quick test:
```bash
cd llama-tests
./test-llama.sh
```

## Common Issues

**GPU not detected:** Verify kernel driver loaded (`lsmod | grep amdgpu`) and device nodes exist (`ls -la /dev/kfd /dev/dri/`)

**Permission denied:** Add user to groups: `sudo usermod -a -G video,render $USER` (logout/login required)

**Out of memory:** Increase shared memory: `--shm-size 16G`

**Slow performance:** Verify GPU usage with `rocm-smi` and ensure GPU layers enabled (`-ngl 99`)

### Model Compatibility Issues

**IBM Granite 4.0 H Tiny + vLLM on ROCm:**

IBM Granite 4.0 H Tiny (and potentially other Granite models) are incompatible with vLLM on ROCm due to flash attention kernel failures. This is caused by:

1. **Non-power-of-2 attention block sizes**: Granite uses 400-token blocks, incompatible with Triton kernels which require power-of-2 sizes
2. **Hybrid MoE architecture incompatibilities**: The model's hybrid Mixture-of-Experts design triggers kernel compilation errors
3. **ROCm flash attention limitations**: Both Triton and ROCm Flash Attention backends fail with `HIP Function Failed - invalid device function`

**Attempted workarounds (all failed):**
- `--enforce-eager` flag
- `VLLM_USE_TRITON_FLASH_ATTN=0`
- `VLLM_ATTENTION_BACKEND=TORCH_SDPA` (not supported on AMD)
- `VLLM_ATTENTION_BACKEND=ROCM_FLASH`
- `VLLM_USE_V1=0` (V0 engine)

**Solution:** Use llama.cpp for Granite models (works perfectly with GGUF format), and use vLLM with tested-compatible models like TinyLlama, Llama-2, Llama-3, Mistral, or Qwen.

See [ROCM_DOCKER_GUIDE.md](ROCM_DOCKER_GUIDE.md) for detailed troubleshooting.

## License

See [LICENSE](LICENSE) file.

---

*Last Updated: October 2025*
