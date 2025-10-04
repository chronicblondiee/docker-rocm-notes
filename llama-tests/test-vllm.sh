#!/bin/bash
# Test script for vLLM Docker Compose deployment
# Tests the vLLM server API endpoints

set -e

# Configuration
COMPOSE_FILE="docker-compose-production.yml"
CONTAINER_NAME="vllm-production"
API_URL="${API_URL:-http://localhost:8001}"
TIMEOUT=300  # seconds to wait for server to start (vLLM takes longer)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "vLLM Docker Compose Test Script"
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
    grep -E "^(MODEL_NAME|VLLM_PORT|GPU_ID|ROCM_ARCH)" .env 2>/dev/null | sed 's/^/   /' || print_info "No vLLM config found in .env"
else
    print_info ".env file not found, using defaults"
fi
echo ""

# Check if container is running
echo "Step 2: Checking vLLM container..."
if docker ps | grep -q "$CONTAINER_NAME"; then
    print_status 0 "Container is running"
else
    print_status 1 "Container is not running"
    echo ""
    print_info "Starting vLLM service..."
    cd ..
    docker-compose -f "$COMPOSE_FILE" up -d
    cd llama-tests
fi
echo ""

# Wait for service to be ready
echo "Step 3: Waiting for vLLM to be ready..."
print_info "This may take several minutes (model download + loading)..."
print_info "Waiting up to ${TIMEOUT}s..."
START_TIME=$(date +%s)
READY=false

while [ $(($(date +%s) - START_TIME)) -lt $TIMEOUT ]; do
    if curl -sf "$API_URL/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 5
    echo -n "."
done
echo ""

if [ "$READY" = true ]; then
    ELAPSED=$(($(date +%s) - START_TIME))
    print_status 0 "Service is ready (took ${ELAPSED}s)"
else
    print_status 1 "Service readiness timeout"
    echo ""
    print_info "Container logs (last 50 lines):"
    cd ..
    docker-compose -f "$COMPOSE_FILE" logs --tail=50
    cd llama-tests
    echo ""
    print_info "The model may still be downloading. Check logs with:"
    echo "   docker-compose -f docker-compose-production.yml logs -f"
    exit 1
fi
echo ""

# Test health endpoint
echo "Step 4: Testing health endpoint..."
if curl -sf "$API_URL/health" > /dev/null; then
    print_status 0 "Health endpoint responding"
else
    print_status 1 "Health endpoint not responding"
    exit 1
fi
echo ""

# Test models endpoint
echo "Step 5: Testing models endpoint..."
MODELS_RESPONSE=$(curl -sf "$API_URL/v1/models" 2>/dev/null)
if [ $? -eq 0 ]; then
    print_status 0 "Models endpoint responding"
    echo "   Available models:"
    echo "$MODELS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for model in data.get('data', []):
        print('   - ' + model.get('id', 'unknown'))
except:
    print('   (Could not parse response)')
" 2>/dev/null || echo "$MODELS_RESPONSE"
else
    print_status 1 "Models endpoint not responding"
fi
echo ""

# Test chat completion endpoint
echo "Step 6: Testing chat completion endpoint..."
print_info "Sending test prompt: 'Say hello in one word'"

# Get model name from response
MODEL_ID=$(echo "$MODELS_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('data', [{}])[0].get('id', 'unknown'))
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

CHAT_RESPONSE=$(curl -sf "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL_ID\",
        \"messages\": [
            {\"role\": \"user\", \"content\": \"Say hello in one word\"}
        ],
        \"temperature\": 0.7,
        \"max_tokens\": 10
    }" 2>/dev/null)

if [ $? -eq 0 ]; then
    print_status 0 "Chat completion endpoint responding"
    echo "   Response:"
    echo "$CHAT_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    content = data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
    print('   ' + content.strip())

    # Show usage stats
    usage = data.get('usage', {})
    if usage:
        print('   Usage: {} prompt + {} completion = {} total tokens'.format(
            usage.get('prompt_tokens', 0),
            usage.get('completion_tokens', 0),
            usage.get('total_tokens', 0)
        ))
except Exception as e:
    print('   (Could not parse response)')
" 2>/dev/null || echo "   $CHAT_RESPONSE"
else
    print_status 1 "Chat completion endpoint failed"
fi
echo ""

# GPU usage check
echo "Step 7: Checking GPU usage..."
if command -v rocm-smi &> /dev/null; then
    print_info "GPU Status:"
    rocm-smi --showuse | grep -E "GPU|%" || rocm-smi
else
    print_info "rocm-smi not available, checking container logs for GPU info"
    cd ..
    docker-compose -f "$COMPOSE_FILE" logs | grep -i "gpu\|rocm\|device\|gfx" | tail -10
    cd llama-tests
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
print_status 0 "vLLM service is running and responding"
print_info "API available at: $API_URL"
print_info "Model: $MODEL_ID"
echo ""
print_info "View logs: docker-compose -f docker-compose-production.yml logs -f"
print_info "Stop service: docker-compose -f docker-compose-production.yml down"
echo ""
echo "Example API calls:"
echo ""
echo "# Chat completion:"
echo "curl $API_URL/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"$MODEL_ID\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "# List models:"
echo "curl $API_URL/v1/models"
echo ""
