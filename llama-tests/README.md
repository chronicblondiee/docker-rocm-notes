# llama.cpp Test Scripts

Test scripts for validating and benchmarking llama.cpp Docker deployments.

## Scripts

### test-llama.sh
Basic functionality test for llama.cpp server.

**What it tests:**
- Configuration validation (.env and models directory)
- Service startup and health checks
- API endpoints (health, models, chat completions)
- GPU detection and usage

**Usage:**
```bash
./test-llama.sh
```

**Output:**
- Step-by-step validation
- API test results with sample responses
- Example curl commands for manual testing

---

### load-test-llama.sh
Load testing script for measuring throughput and response times under concurrent load.

**What it tests:**
- Concurrent request handling
- Response time statistics (min, max, avg, p50, p95, p99)
- Throughput (requests/sec, tokens/sec)
- Success/failure rates

**Configuration (environment variables):**
- `CONCURRENT_REQUESTS` - Number of concurrent requests (default: 10)
- `TOTAL_REQUESTS` - Total requests to send (default: 50)
- `MAX_TOKENS` - Max tokens per response (default: 50)
- `API_URL` - Server URL (default: http://localhost:8000)

**Usage:**
```bash
# Default: 10 concurrent, 50 total requests
./load-test-llama.sh

# Custom configuration
CONCURRENT_REQUESTS=20 TOTAL_REQUESTS=100 ./load-test-llama.sh

# High load test
CONCURRENT_REQUESTS=50 TOTAL_REQUESTS=200 MAX_TOKENS=100 ./load-test-llama.sh
```

**Output:**
- Request summary (success/failure counts)
- Throughput metrics
- Response time percentiles
- Performance assessment

---

### concurrency-test-llama.sh
Analyzes whether requests are processed in parallel or queued sequentially.

**What it tests:**
- Request execution pattern (parallel vs sequential)
- Request overlap analysis
- Timeline visualization
- Speedup vs sequential baseline

**Configuration:**
- `CONCURRENT_REQUESTS` - Requests to launch simultaneously (default: 5)
- `MAX_TOKENS` - Tokens per request (default: 100)
- `API_URL` - Server URL (default: http://localhost:8000)

**Usage:**
```bash
# Default test
./concurrency-test-llama.sh

# Custom test
CONCURRENT_REQUESTS=4 MAX_TOKENS=50 ./concurrency-test-llama.sh
```

**Output:**
- Baseline single-request timing
- Concurrent request timeline (visual)
- Overlap metrics
- Execution pattern verdict (parallel/sequential/mixed)
- Performance recommendations

**Interpreting results:**
- **Sequential**: Requests queued (normal for `PARALLEL_SLOTS=1`)
- **Parallel**: True concurrent execution (requires `PARALLEL_SLOTS>1`)
- **Mixed**: Partial parallelism

---

## Common Workflows

### Initial Setup Validation
```bash
cd llama-tests
./test-llama.sh
```

### Performance Baseline
```bash
# Test with default parallel slots (1)
./concurrency-test-llama.sh
./load-test-llama.sh

# Enable parallel processing
echo "PARALLEL_SLOTS=4" >> ../.env
docker-compose -f ../docker-compose-llamacpp.yml restart

# Test again to see improvement
./concurrency-test-llama.sh
./load-test-llama.sh
```

### Stress Testing
```bash
# Light load
CONCURRENT_REQUESTS=5 TOTAL_REQUESTS=20 ./load-test-llama.sh

# Medium load
CONCURRENT_REQUESTS=10 TOTAL_REQUESTS=50 ./load-test-llama.sh

# Heavy load
CONCURRENT_REQUESTS=20 TOTAL_REQUESTS=100 ./load-test-llama.sh
```

---

## Requirements

- Running llama.cpp server (via docker-compose)
- `curl` - HTTP client
- `python3` - For JSON parsing and statistics
- `docker-compose` - For log inspection

---

## Notes

- All scripts clean up temporary files on exit
- Test results are saved to `*-results-*.txt` (auto-cleaned)
- Scripts use color-coded output for readability
- Set `PARALLEL_SLOTS` in `.env` to enable concurrent request processing
