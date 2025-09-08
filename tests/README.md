# Testing Framework

This directory contains the comprehensive test suite for the Italian Teacher multi-agent framework.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_helpers.py          # Test utility functions
├── unit/                    # Unit tests
│   ├── test_config.py      # Configuration testing
│   └── test_utils.py       # Utility function tests
└── integration/             # Integration tests
    └── test_agent_interaction.py  # Agent interaction tests
```

## Running Tests

### Quick Commands

```bash
# Run all tests
make test

# Run only unit tests (fast)
make test-unit

# Run only integration tests
make test-integration

# Run fast tests (no slow/ML tests)
make test-fast

# Run tests with coverage
make test-cov

# Watch mode (re-run on file changes)
make test-watch
```

### Test Categories

Tests are organized with markers:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, multiple components)
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.ml` - Tests requiring ML models (use `--ml-tests` flag)
- `@pytest.mark.api` - API endpoint tests

### Advanced Usage

```bash
# Run specific test categories
pytest -m unit              # Only unit tests
pytest -m "not slow"        # Skip slow tests  
pytest -m "unit and not ml" # Unit tests, no ML

# Run ML tests (normally skipped)
pytest --ml-tests

# Run with verbose output and stop on first failure
pytest -v -x

# Run specific test file
pytest tests/unit/test_config.py

# Run specific test
pytest tests/unit/test_config.py::TestConfig::test_sample_config_structure
```

## Test Fixtures

Available fixtures (see `conftest.py`):

- `sample_config` - Sample configuration for testing
- `mock_agent` - Mock agent for testing
- `sample_conversation_history` - Sample conversation data
- `sample_training_data` - Sample training data
- `mock_database` - Mock database connection
- `mock_redis` - Mock Redis connection
- `mock_model` - Mock ML model
- `temp_dir` - Temporary directory for file tests

## Writing New Tests

### Unit Test Example

```python
import pytest

class TestMyComponent:
    def test_basic_functionality(self, sample_config):
        # Test basic functionality
        assert sample_config is not None
    
    @pytest.mark.parametrize("input,expected", [
        ("ciao", "hello"),
        ("grazie", "thanks"),
    ])
    def test_translation(self, input, expected):
        # Test with multiple inputs
        result = translate(input)
        assert result == expected
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
class TestAgentWorkflow:
    @pytest.mark.asyncio
    async def test_conversation_flow(self, mock_agent):
        # Test full conversation workflow
        response = await mock_agent.generate_response("Ciao")
        assert response is not None
```

### Mock Usage

```python
from tests.test_helpers import MockAgent, create_test_agents

def test_with_mock_agents():
    agents = create_test_agents()
    marco = agents["marco"]
    assert marco.agent_id == "marco"
```

## Coverage

Generate coverage reports:

```bash
# HTML coverage report (opens in browser)
make test-cov

# Unit test coverage only
make test-cov-unit
```

Coverage reports are generated in `htmlcov/` directory.

## Test Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --strict-markers"
asyncio_mode = "auto"
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "slow: Slow running tests",
    "ml: Tests requiring ML models",
    "api: API tests",
]
```

## Best Practices

1. **Use appropriate markers** - Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

2. **Mock external dependencies** - Use provided fixtures for databases, Redis, models

3. **Test Italian text** - Use `assert_italian_text()` helper for validating Italian responses

4. **Use parametrize for multiple cases** - `@pytest.mark.parametrize` for testing multiple inputs

5. **Async tests** - Use `@pytest.mark.asyncio` for async function tests

6. **Temporary files** - Use `temp_dir` fixture for file operations

7. **Configuration testing** - Use `sample_config` fixture and `create_test_config()` helper

## Continuous Integration

Tests run automatically on:
- Pre-commit hooks (fast tests only)
- CI/CD pipelines (all tests)

Skip slow/ML tests in development:
```bash
make test-fast  # Recommended for development
```