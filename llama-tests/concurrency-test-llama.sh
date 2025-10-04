#!/bin/bash
# Concurrency analysis script for llama.cpp
# Detects if requests are actually running in parallel or queued

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-5}"
MAX_TOKENS="${MAX_TOKENS:-100}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Results file
RESULTS_FILE="./concurrency-test-$$.txt"
trap "rm -f $RESULTS_FILE" EXIT

echo "=========================================="
echo "llama.cpp Concurrency Analysis"
echo "=========================================="
echo ""
echo "This test determines if requests run in parallel or are queued."
echo ""
echo "Configuration:"
echo "  API URL: $API_URL"
echo "  Concurrent requests: $CONCURRENT_REQUESTS"
echo "  Max tokens: $MAX_TOKENS"
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

print_metric() {
    echo -e "${BLUE}▸ $1${NC}"
}

print_analysis() {
    echo -e "${CYAN}→ $1${NC}"
}

# Check if server is running
echo "Step 1: Checking server availability..."
if curl -sf "$API_URL/health" > /dev/null 2>&1; then
    print_status 0 "Server is responding"
else
    print_status 1 "Server is not responding at $API_URL"
    exit 1
fi
echo ""

# Baseline: Single request to measure individual response time
echo "Step 2: Measuring baseline (single request)..."
BASELINE_START=$(date +%s%N)
curl -sf "$API_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 10.\"}],
        \"temperature\": 0.7,
        \"max_tokens\": $MAX_TOKENS
    }" > /dev/null 2>&1
BASELINE_END=$(date +%s%N)
BASELINE_MS=$(((BASELINE_END - BASELINE_START) / 1000000))
print_metric "Baseline response time: ${BASELINE_MS}ms"
echo ""

# Function to send a request and record timing
send_timed_request() {
    local id=$1
    local global_start=$2

    local start_time=$(date +%s%N)
    local start_offset=$(((start_time - global_start) / 1000000))

    curl -sf "$API_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 10.\"}],
            \"temperature\": 0.7,
            \"max_tokens\": $MAX_TOKENS
        }" > /dev/null 2>&1

    local end_time=$(date +%s%N)
    local duration=$(((end_time - start_time) / 1000000))
    local end_offset=$(((end_time - global_start) / 1000000))

    echo "$id|$start_offset|$end_offset|$duration" >> "$RESULTS_FILE"
}

# Test: Launch all requests simultaneously
echo "Step 3: Launching $CONCURRENT_REQUESTS concurrent requests..."
print_info "All requests launched at once. Measuring overlap..."
echo ""

GLOBAL_START=$(date +%s%N)

# Launch all requests in background
for ((i=0; i<CONCURRENT_REQUESTS; i++)); do
    send_timed_request $i $GLOBAL_START &
    echo -n "."
done

# Wait for all to complete
wait
echo ""
echo ""

GLOBAL_END=$(date +%s%N)
TOTAL_TIME=$(((GLOBAL_END - GLOBAL_START) / 1000000))

# Analyze results
echo "Step 4: Analyzing concurrency..."
echo ""

if [ ! -f "$RESULTS_FILE" ]; then
    print_status 1 "No results found"
    exit 1
fi

# Python analysis script
ANALYSIS=$(python3 << 'EOF'
import sys

requests = []

# Read all request data
with open(sys.argv[1], 'r') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) == 4:
            req_id, start_ms, end_ms, duration = map(int, parts)
            requests.append({
                'id': req_id,
                'start': start_ms,
                'end': end_ms,
                'duration': duration
            })

if not requests:
    print("0|0|0|0|0|0")
    sys.exit(0)

# Sort by start time
requests.sort(key=lambda x: x['start'])

# Calculate metrics
total_requests = len(requests)
avg_duration = sum(r['duration'] for r in requests) / total_requests
min_duration = min(r['duration'] for r in requests)
max_duration = max(r['duration'] for r in requests)

# Calculate overlap
# For each request, count how many other requests were running at the same time
overlaps = []
for i, req in enumerate(requests):
    concurrent_count = 0
    for j, other in enumerate(requests):
        if i != j:
            # Check if there's any time overlap
            if not (req['end'] <= other['start'] or req['start'] >= other['end']):
                concurrent_count += 1
    overlaps.append(concurrent_count)

avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
max_overlap = max(overlaps) if overlaps else 0

# Calculate total span
total_span = requests[-1]['end'] - requests[0]['start']

# Detect execution pattern
if avg_overlap >= (total_requests - 1) * 0.8:
    pattern = "parallel"
elif avg_overlap >= 1:
    pattern = "mixed"
else:
    pattern = "sequential"

print(f"{avg_duration:.0f}|{min_duration}|{max_duration}|{avg_overlap:.1f}|{max_overlap}|{total_span}|{pattern}")

# Print timeline
print("\nTIMELINE:", file=sys.stderr)
for req in requests:
    bar_length = 50
    start_pos = int((req['start'] / total_span) * bar_length)
    duration_pos = int((req['duration'] / total_span) * bar_length)
    duration_pos = max(1, duration_pos)

    timeline = [' '] * bar_length
    for p in range(start_pos, min(start_pos + duration_pos, bar_length)):
        timeline[p] = '█'

    print(f"Req {req['id']}: |{''.join(timeline)}| {req['duration']}ms", file=sys.stderr)

EOF
"$RESULTS_FILE" 2>&1)

# Extract metrics from first line, timeline from rest
METRICS=$(echo "$ANALYSIS" | head -1)
TIMELINE=$(echo "$ANALYSIS" | grep "^Req\|^TIMELINE")

IFS='|' read -r AVG_DURATION MIN_DURATION MAX_DURATION AVG_OVERLAP MAX_OVERLAP TOTAL_SPAN PATTERN <<< "$METRICS"

# Display results
echo "=========================================="
echo "Concurrency Analysis Results"
echo "=========================================="
echo ""

echo "Request Timing:"
print_metric "Total wall-clock time: ${TOTAL_TIME}ms"
print_metric "Baseline (single): ${BASELINE_MS}ms"
print_metric "Average duration: ${AVG_DURATION}ms"
print_metric "Min duration: ${MIN_DURATION}ms"
print_metric "Max duration: ${MAX_DURATION}ms"
echo ""

echo "Concurrency Metrics:"
print_metric "Average concurrent requests: ${AVG_OVERLAP}"
print_metric "Maximum concurrent requests: ${MAX_OVERLAP}"
print_metric "Execution pattern: $PATTERN"
echo ""

# Timeline visualization
echo "Timeline Visualization:"
echo "$TIMELINE"
echo ""

# Analysis and verdict
echo "=========================================="
echo "Concurrency Assessment"
echo "=========================================="
echo ""

# Calculate expected time if truly parallel vs sequential
EXPECTED_PARALLEL=$BASELINE_MS
EXPECTED_SEQUENTIAL=$((BASELINE_MS * CONCURRENT_REQUESTS))
ACTUAL=$TOTAL_TIME

print_analysis "Expected if PARALLEL: ~${EXPECTED_PARALLEL}ms (all at once)"
print_analysis "Expected if SEQUENTIAL: ~${EXPECTED_SEQUENTIAL}ms (one after another)"
print_analysis "Actual time: ${ACTUAL}ms"
echo ""

# Determine execution mode
if [ "$PATTERN" = "parallel" ]; then
    print_status 0 "Requests are running CONCURRENTLY"
    echo ""
    print_info "Evidence:"
    echo "  - Average overlap: ${AVG_OVERLAP} (high)"
    echo "  - Total time (~${ACTUAL}ms) close to baseline (${BASELINE_MS}ms)"
    echo "  - Timeline shows significant overlap"
    echo ""
    print_info "Your llama.cpp server handles concurrent requests in parallel."

elif [ "$PATTERN" = "sequential" ]; then
    print_status 1 "Requests are QUEUED (running sequentially)"
    echo ""
    print_info "Evidence:"
    echo "  - Average overlap: ${AVG_OVERLAP} (low/none)"
    echo "  - Total time (~${ACTUAL}ms) close to sequential (${EXPECTED_SEQUENTIAL}ms)"
    echo "  - Timeline shows minimal overlap"
    echo ""
    print_info "Your llama.cpp server is processing requests one at a time."
    echo ""
    echo "Possible reasons:"
    echo "  - llama.cpp typically processes requests sequentially"
    echo "  - Single model instance cannot parallelize generation"
    echo "  - This is normal behavior for most LLM servers"
    echo ""
    echo "To improve throughput:"
    echo "  - Use smaller models for faster individual responses"
    echo "  - Reduce max_tokens to decrease response time"
    echo "  - Consider using vLLM for better concurrent handling"

else
    print_info "Requests show MIXED execution (partial parallelism)"
    echo ""
    print_info "Evidence:"
    echo "  - Average overlap: ${AVG_OVERLAP} (moderate)"
    echo "  - Some requests overlap, others queue"
    echo ""
    print_info "Some parallel processing occurs, but not full concurrency."
fi

echo ""
echo "=========================================="
echo ""

# Performance calculation
if [ "$ACTUAL" -gt 0 ]; then
    THROUGHPUT=$(python3 -c "print(f'{$CONCURRENT_REQUESTS * 1000 / $ACTUAL:.2f}')" 2>/dev/null || echo "N/A")
    print_metric "Throughput: ${THROUGHPUT} requests/second"

    SPEEDUP=$(python3 -c "print(f'{$EXPECTED_SEQUENTIAL / $ACTUAL:.2f}')" 2>/dev/null || echo "N/A")
    print_metric "Speedup vs sequential: ${SPEEDUP}x"
fi

echo ""
