# OpenAI API Compatibility Guide

Complete guide for using OpenAI-compatible APIs with llama.cpp, Ollama, and vLLM on AMD ROCm.

## Table of Contents

- [Overview](#overview)
- [Supported Endpoints](#supported-endpoints)
- [Setup Requirements](#setup-requirements)
- [Tool/Function Calling](#toolfunction-calling)
- [Usage Examples](#usage-examples)
- [Model Recommendations](#model-recommendations)
- [Feature Comparison](#feature-comparison)
- [Troubleshooting](#troubleshooting)

---

## Overview

All three backends (llama.cpp, Ollama, vLLM) provide OpenAI-compatible API endpoints, allowing you to use the official OpenAI Python SDK or any OpenAI-compatible client library with locally hosted models.

### Key Benefits

- **Drop-in replacement** for OpenAI API in existing applications
- **No API keys or rate limits** - fully local inference
- **Privacy-focused** - data never leaves your machine
- **Cost-effective** - no per-token pricing
- **Flexibility** - switch between different backends easily

---

## Supported Endpoints

### Endpoint Support Matrix

| Endpoint | llama.cpp | Ollama | vLLM | Notes |
|----------|-----------|--------|------|-------|
| `POST /v1/chat/completions` | ✅ | ✅ | ✅ | Full support including streaming |
| `POST /v1/completions` | ✅ | ✅ | ✅ | Legacy completion endpoint |
| `POST /v1/embeddings` | ✅ | ✅ | ✅ | Text embeddings generation |
| `GET /v1/models` | ✅ | ✅ | ✅ | List available models |
| **Tool/Function Calling** | ✅ * | ✅ | ✅ | See requirements below |
| **Streaming** | ✅ | ✅ | ✅ | Server-sent events (SSE) |
| **JSON Mode** | ✅ | ✅ | ✅ | Structured output |
| **Vision** | ⚠️ | ✅ | ⚠️ | Model-dependent |
| **Parallel Tool Calls** | ⚠️ | ✅ | ✅ | Optional in llama.cpp |

\* llama.cpp requires `--jinja` flag (enabled by default in our docker-compose setup)

### Base URLs

- **llama.cpp**: `http://localhost:8000/v1`
- **Ollama**: `http://localhost:11434/v1`
- **vLLM**: `http://localhost:8000/v1` (default, configurable via `VLLM_PORT`)

---

## Setup Requirements

### llama.cpp

**Required Configuration:**
- `--jinja` flag must be enabled (✅ already configured in [docker-compose-llamacpp.yml](docker-compose-llamacpp.yml))
- Compatible model with proper chat template

**No additional setup needed** if using our docker-compose files.

### Ollama

**No special configuration required** - OpenAI API works out of the box!

Simply:
1. Start Ollama: `docker-compose -f docker-compose-ollama.yml up -d`
2. Pull a model: `docker exec -it ollama-dev ollama pull llama3.1`
3. Use OpenAI API at `http://localhost:11434/v1`

### vLLM

**Pre-configured** in [docker-compose-vllm.yml](docker-compose-vllm.yml) with OpenAI-compatible server.

Server runs on port `8000` by default (or `VLLM_PORT` from `.env`).

---

## Tool/Function Calling

Tool calling (also known as function calling) allows models to invoke external functions/tools in a structured way.

### Requirements by Backend

#### llama.cpp
- ✅ `--jinja` flag enabled (included in our docker-compose)
- ✅ Compatible model (see [Model Recommendations](#model-recommendations))
- ✅ Q4_K_M or higher quantization (extreme quantization degrades performance)

#### Ollama
- ✅ Compatible model only (no special configuration)
- ✅ Automatic detection and setup

#### vLLM
- ✅ Compatible model (HuggingFace format)
- ✅ Built-in support for most modern models

### Compatible Models for Tool Calling

| Model | Size | llama.cpp | Ollama | vLLM | Notes |
|-------|------|-----------|--------|------|-------|
| **Llama 3.1 / 3.3** | 8B, 70B | ✅ | ✅ | ✅ | **Best overall choice** |
| **Mistral Nemo** | 12B | ✅ | ✅ | ✅ | Excellent performance |
| **Qwen 2.5** | 7B, 14B, 32B | ✅ | ✅ | ✅ | Great multilingual |
| **Hermes 2/3** | 7B, 13B | ✅ | ✅ | ✅ | Function calling focused |
| **Functionary v3.1/v3.2** | 7B | ✅ | ✅ | ✅ | Specialized for tools |
| **Firefunction v2** | 7B | ✅ | ✅ | ✅ | Function calling optimized |
| **Granite 4.0** | Various | ❌ | ❌ | ❌ | Does NOT support tools |
| **Llama 3.2** | 1B, 3B | ❌ | ❌ | ❌ | No tool support |

---

## Usage Examples

### 1. Basic Chat Completion

#### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

#### Python (OpenAI SDK)

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

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

#### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed'
});

const response = await client.chat.completions.create({
  model: 'gpt-3.5-turbo',
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' }
  ],
  temperature: 0.7,
  max_tokens: 100
});

console.log(response.choices[0].message.content);
```

### 2. Streaming Chat

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Count from 1 to 10"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

#### curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Count from 1 to 10"}],
    "stream": true
  }'
```

### 3. Tool/Function Calling

#### Python - Complete Example

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="http://localhost:8000/v1",  # or http://localhost:11434/v1 for Ollama
    api_key="not-needed"
)

# Define tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL, GOOGL"
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]

# Make request with tools
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # Model name doesn't matter for local inference
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco and what's Apple's stock price?"}
    ],
    tools=tools,
    tool_choice="auto"  # Let model decide when to use tools
)

# Check if model wants to call tools
message = response.choices[0].message

if message.tool_calls:
    print("Model wants to call these functions:")
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(f"  - {function_name}({function_args})")

        # In a real application, you would:
        # 1. Execute the actual function
        # 2. Send the result back to the model
        # 3. Get the final response
else:
    print("Response:", message.content)
```

#### curl - Tool Calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "What is the weather in Tokyo?"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

### 4. JSON Mode (Structured Output)

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
        {"role": "user", "content": "List 3 famous scientists with their fields"}
    ],
    response_format={"type": "json_object"}
)

import json
data = json.loads(response.choices[0].message.content)
print(json.dumps(data, indent=2))
```

### 5. Embeddings

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Single text
response = client.embeddings.create(
    model="text-embedding-ada-002",  # Model name can be anything
    input="Hello, world!"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Multiple texts
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=["Hello world", "Goodbye world", "Testing embeddings"]
)

for i, data in enumerate(response.data):
    print(f"Text {i+1} embedding dimension: {len(data.embedding)}")
```

#### curl

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
  }'
```

### 6. List Models

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

models = client.models.list()
for model in models.data:
    print(f"Model: {model.id}")
```

#### curl

```bash
curl http://localhost:8000/v1/models
```

---

## Model Recommendations

### For Tool/Function Calling

**Best Overall (2025):**
1. **Llama 3.1 8B** - Best balance of performance and resource usage
2. **Mistral Nemo 12B** - Excellent tool calling, slightly larger
3. **Qwen 2.5 7B** - Great multilingual support

**High Performance:**
- **Llama 3.3 70B** - Highest quality (requires significant VRAM)
- **Qwen 2.5 32B** - Strong reasoning capabilities

**Specialized:**
- **Functionary v3.2** - Purpose-built for function calling
- **Firefunction v2** - Optimized for tool use

### For General Chat

- **Llama 3.2 3B** - Fast, efficient, good quality (no tool support)
- **Llama 3.1 8B** - Best all-around choice
- **Mistral 7B** - Strong alternative

### For Embeddings

- Models with embedding support (check model documentation)
- Dedicated embedding models for best results

### Quantization Recommendations

| Use Case | Recommended | Notes |
|----------|------------|-------|
| **Tool Calling** | Q4_K_M, Q5_K_M, Q6_K | Avoid extreme quantization |
| **General Chat** | Q4_K_M | Best balance |
| **High Quality** | Q6_K, Q8_0 | Minimal quality loss |
| **Low VRAM** | Q3_K_M, Q4_K_S | Acceptable for chat |

---

## Feature Comparison

### OpenAI API Features

| Feature | llama.cpp | Ollama | vLLM | Implementation Notes |
|---------|-----------|--------|------|---------------------|
| **Chat Completions** | ✅ | ✅ | ✅ | Full support |
| **Streaming** | ✅ | ✅ | ✅ | SSE format |
| **Tool Calling** | ✅ * | ✅ | ✅ | *Requires `--jinja` |
| **Parallel Tools** | ⚠️ | ✅ | ✅ | Optional in llama.cpp |
| **JSON Mode** | ✅ | ✅ | ✅ | response_format support |
| **Vision** | ⚠️ | ✅ | ⚠️ | Model-dependent |
| **Embeddings** | ✅ | ✅ | ✅ | Model must support |
| **System Messages** | ✅ | ✅ | ✅ | Full support |
| **Temperature** | ✅ | ✅ | ✅ | 0.0 - 2.0 |
| **top_p** | ✅ | ✅ | ✅ | Nucleus sampling |
| **max_tokens** | ✅ | ✅ | ✅ | Output length limit |
| **frequency_penalty** | ✅ | ✅ | ✅ | Repetition control |
| **presence_penalty** | ✅ | ✅ | ✅ | Topic diversity |
| **stop** | ✅ | ✅ | ✅ | Custom stop sequences |
| **n** (multiple responses) | ⚠️ | ✅ | ✅ | Limited in llama.cpp |
| **logprobs** | ✅ | ⚠️ | ✅ | Varies by backend |

### Setup Complexity

| Backend | Setup Difficulty | Tool Calling Setup | Notes |
|---------|-----------------|-------------------|-------|
| **Ollama** | ⭐ Easy | None required | Just pull models |
| **llama.cpp** | ⭐⭐ Medium | `--jinja` flag | Pre-configured in our setup |
| **vLLM** | ⭐⭐⭐ Advanced | None required | Needs HuggingFace models |

---

## Troubleshooting

### Tool Calling Not Working

**llama.cpp:**
```bash
# Check if --jinja flag is present
docker exec llama-server ps aux | grep jinja

# Should see: llama-server ... --jinja

# If missing, update docker-compose-llamacpp.yml command section
```

**Solution:** Ensure `--jinja` flag is in the command (already added in our docker-compose).

**Model Compatibility:**
```python
# Test if model supports tools
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's 2+2?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Do math",
            "parameters": {
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"]
            }
        }
    }]
)

# If no tool_calls in response, model doesn't support tools
```

### Connection Refused

```bash
# Check if container is running
docker ps | grep llama-server

# Check logs
docker logs llama-server

# Test health endpoint
curl http://localhost:8000/health

# For Ollama
curl http://localhost:11434/api/version
```

### Model Not Found

**llama.cpp:** Model name doesn't matter - uses MODEL_PATH from environment

**Ollama:**
```bash
# List available models
docker exec ollama-dev ollama list

# Pull missing model
docker exec ollama-dev ollama pull llama3.1
```

**vLLM:** Model must be in MODELS_DIR or be a valid HuggingFace model ID

### Poor Tool Calling Performance

1. **Check quantization:** Use Q4_K_M or higher
2. **Verify model:** Not all models support tools (see compatibility table)
3. **Check logs:** Look for template errors or warnings

```bash
docker logs llama-server | grep -i "tool\|function\|template"
```

### Slow Response Times

1. **Check GPU usage:**
```bash
docker exec llama-server rocm-smi
```

2. **Verify GPU layers:**
```bash
# In .env file
GPU_LAYERS=99  # Should offload all layers
```

3. **Check concurrent requests:**
```bash
# In .env file
PARALLEL_SLOTS=4  # Increase for better concurrency
```

### JSON Mode Not Working

Ensure your prompt explicitly requests JSON output:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You must respond with valid JSON."},
        {"role": "user", "content": "List 3 colors"}
    ],
    response_format={"type": "json_object"}
)
```

---

## Additional Resources

- **llama.cpp Documentation:** https://github.com/ggml-org/llama.cpp
- **Ollama API Docs:** https://github.com/ollama/ollama/blob/main/docs/api.md
- **Ollama OpenAI Compatibility:** https://ollama.com/blog/openai-compatibility
- **vLLM Documentation:** https://docs.vllm.ai/
- **OpenAI API Reference:** https://platform.openai.com/docs/api-reference

---

*Last Updated: October 2025*
