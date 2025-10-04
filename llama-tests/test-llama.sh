#!/bin/bash
# Test script for llama.cpp Docker Compose deployment
# Tests the llama-cpp server API endpoints

set -e

# Configuration
COMPOSE_FILE="docker-compose-llama.yml"
CONTAINER_NAME="llama-server"
API_URL="http://localhost:8000"
TIMEOUT=120  # seconds to wait for server to start

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "llama.cpp Docker Compose Test Script"
echo "=========================================="
echo ""

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if .env file exists
echo "Step 1: Checking configuration..."
if [ -f .env ]; then
    print_status 0 ".env file found"
    echo "   Environment variables:"
    grep -E "^(MODELS_DIR|MODEL_PATH|LLAMA_PORT|GPU_ID)" .env | sed 's/^/   /'
else
    print_info ".env file not found, using defaults"
    print_info "Default: MODELS_DIR=./models, MODEL_PATH=/data/model.gguf"
fi
echo ""

# Check if models directory exists
echo "Step 2: Checking models directory..."
MODELS_DIR=$(grep "^MODELS_DIR=" .env 2>/dev/null | cut -d'=' -f2 || echo "./models")
if [ -d "$MODELS_DIR" ]; then
    print_status 0 "Models directory exists: $MODELS_DIR"
    MODEL_COUNT=$(find "$MODELS_DIR" -name "*.gguf" 2>/dev/null | wc -l)
    if [ $MODEL_COUNT -gt 0 ]; then
        print_status 0 "Found $MODEL_COUNT GGUF model(s)"
        find "$MODELS_DIR" -name "*.gguf" -exec basename {} \; | sed 's/^/   - /'
    else
        print_status 1 "No GGUF models found in $MODELS_DIR"
        echo "   Please add a .gguf model file to $MODELS_DIR"
        exit 1
    fi
else
    print_status 1 "Models directory not found: $MODELS_DIR"
    exit 1
fi
echo ""

# Start the service
echo "Step 3: Starting llama.cpp service..."
print_info "Running: docker-compose -f $COMPOSE_FILE up -d"
if docker-compose -f "$COMPOSE_FILE" up -d; then
    print_status 0 "Service started"
else
    print_status 1 "Failed to start service"
    exit 1
fi
echo ""

# Wait for service to be healthy
echo "Step 4: Waiting for service to be ready..."
print_info "Waiting up to ${TIMEOUT}s for health check..."
START_TIME=$(date +%s)
HEALTHY=false

while [ $(($(date +%s) - START_TIME)) -lt $TIMEOUT ]; do
    if docker inspect --format='{{.State.Health.Status}}' "$CONTAINER_NAME" 2>/dev/null | grep -q "healthy"; then
        HEALTHY=true
        break
    fi
    sleep 2
    echo -n "."
done
echo ""

if [ "$HEALTHY" = true ]; then
    ELAPSED=$(($(date +%s) - START_TIME))
    print_status 0 "Service is healthy (took ${ELAPSED}s)"
else
    print_status 1 "Service health check timeout"
    echo ""
    print_info "Container logs:"
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    exit 1
fi
echo ""

# Test health endpoint
echo "Step 5: Testing health endpoint..."
if curl -sf "$API_URL/health" > /dev/null; then
    print_status 0 "Health endpoint responding"
else
    print_status 1 "Health endpoint not responding"
    exit 1
fi
echo ""

# Test models endpoint
echo "Step 6: Testing models endpoint..."
MODELS_RESPONSE=$(curl -sf "$API_URL/v1/models" 2>/dev/null)
if [ $? -eq 0 ]; then
    print_status 0 "Models endpoint responding"
    echo "$MODELS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"
else
    print_status 1 "Models endpoint not responding"
fi
echo ""

# Test chat completion endpoint
echo "Step 7: Testing chat completion endpoint..."
print_info "Sending test prompt: 'Say hello in one word'"
CHAT_RESPONSE=$(curl -sf "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "user", "content": "Say hello in one word"}
        ],
        "temperature": 0.7,
        "max_tokens": 10
    }' 2>/dev/null)

if [ $? -eq 0 ]; then
    print_status 0 "Chat completion endpoint responding"
    echo "   Response:"
    echo "$CHAT_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
    print('   ' + content.strip())
except:
    print('   (Could not parse response)')
" 2>/dev/null || echo "   $CHAT_RESPONSE"
else
    print_status 1 "Chat completion endpoint failed"
fi
echo ""

# GPU usage check
echo "Step 8: Checking GPU usage..."
if command -v rocm-smi &> /dev/null; then
    print_info "GPU Status:"
    rocm-smi --showuse | grep -E "GPU|%" || rocm-smi
else
    print_info "rocm-smi not available on host, checking container logs for GPU info"
    docker-compose -f "$COMPOSE_FILE" logs | grep -i "gpu\|rocm\|device" | tail -5
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
print_status 0 "llama.cpp service is running and responding"
print_info "API available at: $API_URL"
print_info "View logs: docker-compose -f $COMPOSE_FILE logs -f"
print_info "Stop service: docker-compose -f $COMPOSE_FILE down"
echo ""
echo "Example API calls:"
echo ""
echo "# Chat completion:"
echo "curl $API_URL/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "# Text completion:"
echo "curl $API_URL/v1/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"prompt\": \"Once upon a time\", \"max_tokens\": 50}'"
echo ""
