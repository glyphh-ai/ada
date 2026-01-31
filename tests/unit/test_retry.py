"""
Unit tests for retry and circuit breaker utilities.
"""

import asyncio
import pytest
import time

from shared.retry import (
    retry_with_backoff,
    CircuitBreaker,
    CircuitState,
    RetryConfig,
    circuit_breaker,
)


class TestRetryConfig:
    """Tests for RetryConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RetryConfig()
        
        assert config.max_retries == 5
        assert config.base_delay_ms == 100
        assert config.max_delay_ms == 1600
        assert config.exponential_base == 2.0
    
    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        config = RetryConfig(base_delay_ms=100, exponential_base=2.0)
        
        assert config.get_delay(0) == 0.1  # 100ms
        assert config.get_delay(1) == 0.2  # 200ms
        assert config.get_delay(2) == 0.4  # 400ms
        assert config.get_delay(3) == 0.8  # 800ms
    
    def test_get_delay_capped(self):
        """Test that delay is capped at max_delay_ms."""
        config = RetryConfig(base_delay_ms=100, max_delay_ms=500)
        
        # Attempt 10 would be 100 * 2^10 = 102400ms, but capped at 500ms
        assert config.get_delay(10) == 0.5


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""
    
    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call without retry."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await successful_func()
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test retry on transient failure."""
        call_count = 0
        
        @retry_with_backoff(max_retries=3, base_delay_ms=10)
        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Transient error")
            return "success"
        
        result = await flaky_func()
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, base_delay_ms=10)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent error")
        
        with pytest.raises(ValueError) as exc_info:
            await always_fails()
        
        assert "Permanent error" in str(exc_info.value)
        assert call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_specific_exception_types(self):
        """Test retry only on specific exception types."""
        call_count = 0
        
        @retry_with_backoff(
            max_retries=3,
            base_delay_ms=10,
            retryable_exceptions=[ValueError],
        )
        async def specific_error():
            nonlocal call_count
            call_count += 1
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            await specific_error()
        
        # Should not retry on TypeError
        assert call_count == 1


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""
    
    def test_initial_state_closed(self):
        """Test that circuit starts in closed state."""
        breaker = CircuitBreaker(name="test")
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True
    
    def test_opens_after_failures(self):
        """Test that circuit opens after threshold failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False
    
    def test_success_resets_failure_count(self):
        """Test that success resets failure count."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)
        
        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        
        # Record success
        breaker.record_success()
        
        # Should still be closed
        assert breaker.state == CircuitState.CLOSED
        
        # Need 3 more failures to open
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED
    
    def test_half_open_after_timeout(self):
        """Test transition to half-open after recovery timeout."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0.1,  # 100ms
        )
        
        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.allow_request() is True
    
    def test_closes_after_successes_in_half_open(self):
        """Test that circuit closes after successes in half-open."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            success_threshold=2,
            recovery_timeout=0.01,
        )
        
        # Open and wait for half-open
        breaker.record_failure()
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Record successes
        breaker.record_success()
        breaker.record_success()
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_reopens_on_failure_in_half_open(self):
        """Test that circuit reopens on failure in half-open."""
        breaker = CircuitBreaker(
            name="test",
            failure_threshold=1,
            recovery_timeout=0.01,
        )
        
        # Open and wait for half-open
        breaker.record_failure()
        time.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Fail in half-open
        breaker.record_failure()
        
        assert breaker.state == CircuitState.OPEN
    
    def test_reset(self):
        """Test manual circuit reset."""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        
        # Open the circuit
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.allow_request() is True
    
    def test_get_stats(self):
        """Test getting circuit breaker statistics."""
        breaker = CircuitBreaker(name="test")
        
        breaker.record_failure()
        breaker.record_failure()
        
        stats = breaker.get_stats()
        
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker decorator."""
    
    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Test decorator with successful call."""
        breaker = CircuitBreaker(name="test")
        
        @circuit_breaker(breaker)
        async def successful_func():
            return "success"
        
        result = await successful_func()
        
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_decorator_records_failure(self):
        """Test that decorator records failures."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        
        @circuit_breaker(breaker)
        async def failing_func():
            raise ValueError("Error")
        
        with pytest.raises(ValueError):
            await failing_func()
        
        with pytest.raises(ValueError):
            await failing_func()
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_decorator_rejects_when_open(self):
        """Test that decorator rejects calls when circuit is open."""
        breaker = CircuitBreaker(name="test", failure_threshold=1)
        
        @circuit_breaker(breaker)
        async def func():
            return "success"
        
        # Open the circuit
        breaker.record_failure()
        
        with pytest.raises(Exception) as exc_info:
            await func()
        
        assert "open" in str(exc_info.value).lower()
