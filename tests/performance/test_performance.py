"""
Performance tests for neuron client and server.

Tests response times, throughput, and performance characteristics.
"""
import pytest
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@pytest.mark.slow
@pytest.mark.integration
def test_health_endpoint_response_time(server_url, server_available):
    """Test health endpoint responds quickly (< 100ms)."""
    measurements = []

    for _ in range(10):
        start = time.time()
        response = requests.get(f"{server_url}/health")
        elapsed = (time.time() - start) * 1000  # Convert to ms

        assert response.status_code == 200
        measurements.append(elapsed)

    avg_time = sum(measurements) / len(measurements)
    max_time = max(measurements)

    # Health check should be fast
    assert avg_time < 100, f"Average response time {avg_time:.2f}ms exceeds 100ms"
    assert max_time < 200, f"Max response time {max_time:.2f}ms exceeds 200ms"


@pytest.mark.slow
@pytest.mark.integration
def test_challenge_token_generation_performance(server_url, server_available):
    """Test challenge token generation is fast."""
    measurements = []

    for _ in range(20):
        start = time.time()
        response = requests.get(f"{server_url}/api/v1/auth/challenge")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        measurements.append(elapsed)

    avg_time = sum(measurements) / len(measurements)

    # Challenge generation should be quick (UUID generation + storage)
    assert avg_time < 150, f"Average time {avg_time:.2f}ms exceeds 150ms"


@pytest.mark.slow
@pytest.mark.integration
def test_observation_retrieval_performance(server_url, server_available):
    """Test observation retrieval is fast."""
    measurements = []

    for _ in range(15):
        start = time.time()
        response = requests.get(f"{server_url}/api/v1/observation/get")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        measurements.append(elapsed)

    avg_time = sum(measurements) / len(measurements)

    # Database query should be fast with in-memory SQLite
    assert avg_time < 200, f"Average time {avg_time:.2f}ms exceeds 200ms"


@pytest.mark.slow
@pytest.mark.integration
def test_multiple_endpoints_performance(server_url, server_available):
    """Test that multiple different endpoints all respond quickly."""
    endpoints = [
        "/health",
        "/",
        "/api/v1/auth/challenge",
        "/api/v1/observation/get",
    ]

    for endpoint in endpoints:
        start = time.time()
        response = requests.get(f"{server_url}{endpoint}")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 200, f"Endpoint {endpoint} took {elapsed:.2f}ms (> 200ms)"


@pytest.mark.slow
@pytest.mark.integration
def test_client_publish_method_performance(client_instance):
    """Test client publish method executes quickly (excluding network)."""
    topic = "perf-test-topic"

    measurements = []

    for i in range(10):
        start = time.time()
        # This will be rate-limited after first call, so measure that too
        result = client_instance.publish(
            topic=topic if i == 0 else f"{topic}-{i}",  # Different topics to avoid rate limit
            data=f"{i * 100}.00",
            observationTime="2025-01-01T00:00:00Z",
            observationHash=f"hash-{i}",
            isPrediction=True
        )
        elapsed = (time.time() - start) * 1000

        measurements.append(elapsed)

    avg_time = sum(measurements) / len(measurements)

    # Client method should be quick (even if auth fails)
    assert avg_time < 500, f"Average time {avg_time:.2f}ms exceeds 500ms"


@pytest.mark.slow
@pytest.mark.integration
def test_sequential_requests_performance(server_url, server_available):
    """Test that sequential requests maintain consistent performance."""
    num_requests = 30
    measurements = []

    for i in range(num_requests):
        start = time.time()
        response = requests.get(f"{server_url}/health")
        elapsed = (time.time() - start) * 1000

        assert response.status_code == 200
        measurements.append(elapsed)

    # Check for performance degradation over time
    first_10_avg = sum(measurements[:10]) / 10
    last_10_avg = sum(measurements[-10:]) / 10

    # Performance should not degrade significantly
    degradation = (last_10_avg - first_10_avg) / first_10_avg * 100

    assert degradation < 50, f"Performance degraded by {degradation:.1f}% over {num_requests} requests"


@pytest.mark.slow
@pytest.mark.integration
def test_challenge_token_uniqueness_at_scale(server_url, server_available):
    """Test that many challenge tokens are generated quickly and remain unique."""
    num_tokens = 50
    start = time.time()

    tokens = set()
    for _ in range(num_tokens):
        response = requests.get(f"{server_url}/api/v1/auth/challenge")
        assert response.status_code == 200

        token = response.json()["challenge"]
        tokens.add(token)

    elapsed = time.time() - start

    # All tokens should be unique
    assert len(tokens) == num_tokens, f"Only {len(tokens)}/{num_tokens} unique tokens"

    # Should complete in reasonable time
    tokens_per_second = num_tokens / elapsed
    assert tokens_per_second > 10, f"Only {tokens_per_second:.1f} tokens/sec (expected > 10)"


@pytest.mark.slow
@pytest.mark.integration
def test_server_handles_rapid_requests(server_url, server_available):
    """Test server can handle rapid successive requests."""
    num_requests = 25
    start = time.time()

    for _ in range(num_requests):
        response = requests.get(f"{server_url}/health")
        assert response.status_code == 200

    elapsed = time.time() - start
    requests_per_second = num_requests / elapsed

    # Should handle at least 20 requests per second
    assert requests_per_second > 20, f"Only {requests_per_second:.1f} req/sec (expected > 20)"


@pytest.mark.slow
@pytest.mark.integration
def test_client_checkin_performance(client_instance):
    """Test client checkin method performance."""
    measurements = []

    for _ in range(5):  # Fewer iterations since checkin does more work
        start = time.time()
        result = client_instance.checkin()
        elapsed = (time.time() - start) * 1000

        assert isinstance(result, dict)
        measurements.append(elapsed)

    avg_time = sum(measurements) / len(measurements)

    # Checkin involves health check + optional other calls
    assert avg_time < 1000, f"Average checkin time {avg_time:.2f}ms exceeds 1000ms"
