#!/bin/bash
# Load test script for llama.cpp chat completions endpoint
# Tests concurrent request handling and throughput

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-10}"
TOTAL_REQUESTS="${TOTAL_REQUESTS:-50}"
MAX_TOKENS="${MAX_TOKENS:-50}"
TEMPERATURE="${TEMPERATURE:-0.7}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Results file in current directory (not temp)
RESULTS_FILE="./load-test-results-$$.txt"
trap "rm -f $RESULTS_FILE" EXIT

echo "=========================================="
echo "llama.cpp Load Test Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Concurrent requests: $CONCURRENT_REQUESTS"
echo "  Total requests: $TOTAL_REQUESTS"
echo "  Max tokens per request: $MAX_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo ""

# Test prompts
PROMPTS=(
    "Explain quantum computing in one sentence."
    "What is the capital of France?"
    "Write a haiku about coding."
    "What are the three laws of robotics?"
    "Explain recursion briefly."
    "What is the speed of light?"
    "Name three programming languages."
    "What is artificial intelligence?"
    "Explain what HTTP stands for."
    "What is the meaning of life?"
)

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

print_metric() {
    echo -e "${BLUE}▸ $1${NC}"
}

# Check if server is running
echo "Step 1: Checking server availability..."
if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    print_status 0 "Server is responding"
else
    print_status 1 "Server is not responding at $API_URL"
    echo "   Please start the server first: docker-compose -f docker-compose-llamacpp.yml up -d"
    exit 1
fi
echo ""

# Function to send a single request
send_request() {
    local id=$1
    local prompt="${PROMPTS[$((RANDOM % ${#PROMPTS[@]}))]}"
    local start_time=$(date +%s%N)

    local response=$(curl -sf "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}],
            \"temperature\": $TEMPERATURE,
            \"max_tokens\": $MAX_TOKENS
        }" 2>/dev/null)

    local end_time=$(date +%s%N)
    local duration=$(((end_time - start_time) / 1000000)) # Convert to milliseconds

    if [ $? -eq 0 ] && [ -n "$response" ]; then
        # Extract tokens from response if available
        local completion_tokens=$(echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null || echo 0)

        echo "$id|$duration|$completion_tokens|success" >> "$RESULTS_FILE"
        echo -n "."
    else
        echo "$id|$duration|0|failed" >> "$RESULTS_FILE"
        echo -n "!"
    fi
}

# Warm-up request
echo "Step 2: Sending warm-up request..."
WARMUP_START=$(date +%s%N)
curl -sf "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}],
        \"temperature\": 0.7,
        \"max_tokens\": 5
    }" > /dev/null 2>&1
WARMUP_END=$(date +%s%N)
WARMUP_TIME=$(((WARMUP_END - WARMUP_START) / 1000000))
print_status 0 "Warm-up completed (${WARMUP_TIME}ms)"
echo ""

# Run load test
echo "Step 3: Running load test..."
print_info "Sending $TOTAL_REQUESTS requests with $CONCURRENT_REQUESTS concurrent connections"
echo ""

TEST_START=$(date +%s%N)

# Send requests in batches
for ((i=0; i<TOTAL_REQUESTS; i+=CONCURRENT_REQUESTS)); do
    batch_size=$CONCURRENT_REQUESTS
    if [ $((i + batch_size)) -gt $TOTAL_REQUESTS ]; then
        batch_size=$((TOTAL_REQUESTS - i))
    fi

    # Launch batch of concurrent requests
    for ((j=0; j<batch_size; j++)); do
        send_request $((i + j)) &
    done

    # Wait for batch to complete
    wait
done

echo ""
TEST_END=$(date +%s%N)
TOTAL_TIME=$(((TEST_END - TEST_START) / 1000000000)) # Convert to seconds
echo ""

# Analyze results
echo "Step 4: Analyzing results..."
echo ""

if [ ! -f "$RESULTS_FILE" ]; then
    print_status 1 "No results found"
    exit 1
fi

# Calculate statistics
STATS=$(python3 << 'EOF'
import sys

results = []
successful = 0
failed = 0
total_tokens = 0

try:
    with open(sys.argv[1], 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 4:
                req_id, duration, tokens, status = parts
                duration_ms = int(duration)
                tokens = int(tokens)

                if status == 'success':
                    successful += 1
                    results.append(duration_ms)
                    total_tokens += tokens
                else:
                    failed += 1
except Exception as e:
    print(f"Error reading results: {e}", file=sys.stderr)

if results:
    results.sort()
    count = len(results)
    avg = sum(results) / count
    min_val = results[0]
    max_val = results[-1]
    p50 = results[int(count * 0.50)]
    p95 = results[int(count * 0.95)]
    p99 = results[int(count * 0.99)] if count > 1 else results[-1]

    print(f"{successful}")
    print(f"{failed}")
    print(f"{min_val:.0f}")
    print(f"{max_val:.0f}")
    print(f"{avg:.0f}")
    print(f"{p50:.0f}")
    print(f"{p95:.0f}")
    print(f"{p99:.0f}")
    print(f"{total_tokens}")
else:
    print("0\n0\n0\n0\n0\n0\n0\n0\n0")
EOF
"$RESULTS_FILE")

read -r SUCCESSFUL FAILED MIN_MS MAX_MS AVG_MS P50_MS P95_MS P99_MS TOTAL_TOKENS <<< "$STATS"

# Calculate throughput
REQUESTS_PER_SEC=$(python3 -c "print(f'{$TOTAL_REQUESTS / $TOTAL_TIME:.2f}')" 2>/dev/null || echo "N/A")
TOKENS_PER_SEC=$(python3 -c "print(f'{$TOTAL_TOKENS / $TOTAL_TIME:.2f}')" 2>/dev/null || echo "N/A")

# Display results
echo "=========================================="
echo "Load Test Results"
echo "=========================================="
echo ""
echo "Request Summary:"
print_metric "Total requests: $TOTAL_REQUESTS"
print_metric "Successful: $SUCCESSFUL ($(python3 -c "print(f'{$SUCCESSFUL*100/$TOTAL_REQUESTS:.1f}')" 2>/dev/null || echo "N/A")%)"
print_metric "Failed: $FAILED ($(python3 -c "print(f'{$FAILED*100/$TOTAL_REQUESTS:.1f}')" 2>/dev/null || echo "N/A")%)"
print_metric "Total time: ${TOTAL_TIME}s"
echo ""

echo "Throughput:"
print_metric "Requests/sec: $REQUESTS_PER_SEC"
print_metric "Tokens/sec: $TOKENS_PER_SEC"
print_metric "Total tokens generated: $TOTAL_TOKENS"
echo ""

echo "Response Times (milliseconds):"
print_metric "Min: ${MIN_MS}ms"
print_metric "Max: ${MAX_MS}ms"
print_metric "Average: ${AVG_MS}ms"
print_metric "Median (P50): ${P50_MS}ms"
print_metric "P95: ${P95_MS}ms"
print_metric "P99: ${P99_MS}ms"
echo ""

# Performance assessment
echo "Performance Assessment:"
if [ "$FAILED" -eq 0 ]; then
    print_status 0 "No failed requests"
else
    print_status 1 "$FAILED failed requests"
fi

if [ $(echo "$AVG_MS < 5000" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
    print_status 0 "Average response time is good (<5s)"
elif [ $(echo "$AVG_MS < 10000" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
    print_info "Average response time is acceptable (<10s)"
else
    print_info "Average response time is slow (>10s)"
fi

echo ""
echo "=========================================="
echo ""
echo "Tips for improving performance:"
echo "  - Increase GPU layers: GPU_LAYERS=99 in .env"
echo "  - Use quantized models (Q4_K_M or Q5_K_M)"
echo "  - Reduce context size if not needed: CONTEXT_SIZE=2048"
echo "  - Monitor GPU usage: rocm-smi"
echo "  - Check container logs: docker-compose -f docker-compose-llamacpp.yml logs"
echo ""
