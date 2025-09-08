# Testing Architecture Guide

This document explains the testing framework architecture, different test types, fixtures, and how they all work together in the Italian Teacher project.

## Test Types & Philosophy

### 1. Unit Tests (`@pytest.mark.unit`)

**Purpose**: Test individual components in isolation
**Speed**: Very fast (< 1ms per test)
**Dependencies**: None (fully mocked)

```python
class TestConfig:
    def test_sample_config_structure(self, sample_config):
        """Test that sample config has required structure."""
        assert "app" in sample_config
        assert "models" in sample_config
        assert "agents" in sample_config
```

**Characteristics**:
- Test single functions or methods
- No external dependencies (database, network, files)
- Use mocks for all external interactions
- Should run in any order
- Deterministic results

### 2. Integration Tests (`@pytest.mark.integration`)

**Purpose**: Test how components work together
**Speed**: Slower (10-100ms per test)
**Dependencies**: Multiple components, mock external services

```python
@pytest.mark.integration
class TestAgentCoordination:
    @pytest.mark.asyncio
    async def test_basic_agent_response_flow(self, mock_agent):
        """Test basic agent response generation flow."""
        user_message = "Ciao! Come stai?"
        response = await mock_agent.generate_response(user_message)
        assert response == "Ciao! Sto bene, grazie! E tu come stai?"
```

**Characteristics**:
- Test interactions between 2-3 components
- Mock external services (but not internal components)
- May involve async operations
- Test workflows and data flow

### 3. Slow Tests (`@pytest.mark.slow`)

**Purpose**: Long-running tests (file I/O, complex operations)
**Speed**: Very slow (> 1 second)
**When to use**: Large data processing, model training simulation

```python
@pytest.mark.slow
class TestEndToEndWorkflow:
    def test_complete_training_pipeline(self):
        """Test entire training pipeline (slow)."""
        # This might take 10+ seconds
        pass
```

### 4. ML Tests (`@pytest.mark.ml`)

**Purpose**: Tests requiring actual ML models
**Speed**: Very slow (model loading/inference)
**Special**: Requires `--ml-tests` flag to run

```python
@pytest.mark.ml
class TestModelIntegration:
    def test_real_model_inference(self):
        """Test with actual transformer model."""
        # Only runs with: pytest --ml-tests
        pass
```

### 5. API Tests (`@pytest.mark.api`)

**Purpose**: Test HTTP endpoints and API interactions
**Dependencies**: FastAPI test client, mock services

```python
@pytest.mark.api
class TestAPIEndpoints:
    def test_chat_endpoint(self, client):
        response = client.post("/chat", json={"message": "Ciao"})
        assert response.status_code == 200
```

## Fixtures Explained

### What are Fixtures?

Fixtures are **reusable setup code** that prepare test data, mock objects, or test environments. They're defined in `conftest.py` and automatically available to all tests.

### Core Data Fixtures

#### `sample_config`
```python
@pytest.fixture
def sample_config():
    return {
        "app": {"name": "Italian Teacher Test", "version": "0.1.0"},
        "models": {"base_model": "microsoft/DialoGPT-small", "device": "cpu"},
        "agents": {
            "marco": {"personality": "friendly_conversationalist"},
            # ... more agents
        }
    }
```

**Used for**: Testing configuration loading, agent initialization, model setup
**Test types**: Unit, Integration
**Example usage**:
```python
def test_agent_creation(self, sample_config):
    agent_config = sample_config["agents"]["marco"]
    agent = Agent(agent_config)
    assert agent.personality == "friendly_conversationalist"
```

#### `sample_conversation_history`
```python
@pytest.fixture
def sample_conversation_history():
    return [
        {"role": "user", "content": "Ciao! Come stai?", "timestamp": "2024-01-01T10:00:00Z"},
        {"role": "marco", "content": "Ciao! Sto bene, grazie! E tu?", "timestamp": "2024-01-01T10:00:05Z"},
        # ... more messages
    ]
```

**Used for**: Testing conversation management, context handling, message processing
**Test types**: Unit, Integration
**Example usage**:
```python
def test_conversation_context(self, sample_conversation_history):
    context_manager = ConversationManager()
    context_manager.load_history(sample_conversation_history)
    assert len(context_manager.get_recent_messages(2)) == 2
```

#### `sample_training_data`
```python
@pytest.fixture
def sample_training_data():
    return [
        {"input": "Come si dice 'hello' in italiano?", "output": "Si dice 'ciao' o 'salve'."},
        # ... more training examples
    ]
```

**Used for**: Testing data processing, training pipelines, model fine-tuning
**Test types**: Unit, ML
**Example usage**:
```python
def test_data_preprocessing(self, sample_training_data):
    processor = DataProcessor()
    processed = processor.prepare_training_data(sample_training_data)
    assert len(processed) == len(sample_training_data)
```

### Mock Fixtures

#### `mock_agent`
```python
@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.agent_id = "test_agent"
    agent.personality = "test_personality"
    agent.generate_response = AsyncMock(return_value="Test response")
    agent.is_available = Mock(return_value=True)
    return agent
```

**Used for**: Testing agent coordination without real agent implementations
**Test types**: Unit, Integration
**Why mock**: Real agents require ML models (slow), this focuses on coordination logic

**Example usage**:
```python
@pytest.mark.asyncio
async def test_coordinator_selects_agent(self, mock_agent):
    coordinator = Coordinator()
    coordinator.register_agent(mock_agent)
    selected = coordinator.select_best_agent("Ciao!")
    assert selected == mock_agent
```

#### `mock_database`
```python
@pytest.fixture
def mock_database():
    db = Mock()
    db.execute = AsyncMock()
    db.fetch_all = AsyncMock(return_value=[])
    db.fetch_one = AsyncMock(return_value=None)
    return db
```

**Used for**: Testing database interactions without actual database
**Test types**: Unit, Integration  
**Why mock**: No database setup required, tests focus on business logic

**Example usage**:
```python
async def test_save_conversation(self, mock_database):
    service = ConversationService(mock_database)
    await service.save_message("user", "Ciao")
    mock_database.execute.assert_called_once()
```

#### `mock_redis`
```python
@pytest.fixture
def mock_redis():
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    return redis
```

**Used for**: Testing caching and session management
**Test types**: Unit, Integration
**Why mock**: No Redis server required, tests focus on cache logic

#### `mock_model`
```python
@pytest.fixture
def mock_model():
    model = Mock()
    model.generate = Mock(return_value=["Generated text response"])
    model.tokenizer = Mock()
    return model
```

**Used for**: Testing ML model integration without loading actual models
**Test types**: Unit, Integration
**Why mock**: Real models are huge (GBs), slow to load, this tests the integration layer

### Utility Fixtures

#### `temp_dir`
```python
@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)  # Cleanup after test
```

**Used for**: Testing file operations, configuration files, data export
**Test types**: Unit, Integration
**Why needed**: Safe isolated environment for file tests

**Example usage**:
```python
def test_config_file_creation(self, temp_dir):
    config_file = temp_dir / "test_config.yaml"
    ConfigManager.save_config(config, config_file)
    assert config_file.exists()
    # temp_dir automatically cleaned up after test
```

## How Fixtures Work Together

### Test Dependency Chain

```python
def test_complete_workflow(self, sample_config, mock_agent, mock_database, temp_dir):
    """Example showing multiple fixtures working together."""
    
    # 1. sample_config provides agent configuration
    agent_config = sample_config["agents"]["marco"]
    
    # 2. mock_agent simulates agent behavior
    mock_agent.personality = agent_config["personality"]
    
    # 3. mock_database handles data persistence
    service = ConversationService(mock_database)
    
    # 4. temp_dir provides safe file operations
    log_file = temp_dir / "conversation.log"
    
    # Test the complete workflow
    # ... test logic here
```

### Fixture Scopes

```python
@pytest.fixture(scope="session")  # Created once per test session
def expensive_setup():
    return setup_expensive_resource()

@pytest.fixture(scope="function")  # Created for each test (default)
def fresh_data():
    return create_test_data()
```

**Scopes**:
- `function` (default): New instance per test
- `class`: Shared within test class
- `module`: Shared within test file
- `session`: Shared across entire test run

## Test Organization Strategy

### Directory Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_helpers.py          # Utility functions
├── unit/                    # Fast, isolated tests
│   ├── test_config.py      # Configuration logic
│   ├── test_utils.py       # Utility functions
│   └── test_agents.py      # Individual agent logic
└── integration/             # Component interaction tests
    ├── test_agent_interaction.py  # Agent coordination
    ├── test_api.py              # API endpoints
    └── test_workflows.py       # End-to-end workflows
```

### When to Use Each Test Type

| Test Type | Use When | Fixtures Commonly Used |
|-----------|----------|----------------------|
| **Unit** | Testing single function/class | `sample_config`, `temp_dir`, simple mocks |
| **Integration** | Testing 2-3 components together | `mock_agent`, `mock_database`, `sample_conversation_history` |
| **Slow** | File I/O, complex processing | `temp_dir`, `sample_training_data` |
| **ML** | Real model testing | `sample_config`, `sample_training_data` |
| **API** | HTTP endpoint testing | `mock_database`, `mock_redis`, API client fixtures |

## Practical Examples

### Unit Test Example
```python
class TestAgent:
    def test_agent_initialization(self, sample_config):
        """Test creating an agent with configuration."""
        config = sample_config["agents"]["marco"]
        agent = Agent(agent_id="marco", config=config)
        
        assert agent.agent_id == "marco"
        assert agent.personality == "friendly_conversationalist"
        # Fast, isolated, no external dependencies
```

### Integration Test Example  
```python
@pytest.mark.integration
class TestConversationFlow:
    @pytest.mark.asyncio
    async def test_user_agent_interaction(self, mock_agent, mock_database):
        """Test complete user-agent conversation flow."""
        # Setup components
        coordinator = Coordinator()
        coordinator.register_agent(mock_agent)
        conversation_service = ConversationService(mock_database)
        
        # Test interaction
        user_input = "Ciao! Come stai?"
        selected_agent = coordinator.select_agent(user_input)
        response = await selected_agent.generate_response(user_input)
        await conversation_service.save_interaction(user_input, response)
        
        # Verify workflow
        assert selected_agent == mock_agent
        mock_database.execute.assert_called()
        # Tests multiple components working together
```

### ML Test Example
```python
@pytest.mark.ml
class TestRealModelInference:
    def test_model_generates_italian_response(self, sample_config):
        """Test actual model generates Italian text."""
        # This only runs with --ml-tests flag
        model = load_model(sample_config["models"]["base_model"])
        response = model.generate("Come stai?")
        
        assert_italian_text(response)  # Helper function
        # Slow, requires actual model, optional
```

## Best Practices

### 1. Fixture Design
- **Single Responsibility**: Each fixture does one thing
- **Predictable Data**: Same data every time (deterministic)
- **Minimal Setup**: Only what's needed for tests
- **Automatic Cleanup**: Use `yield` for teardown

### 2. Test Design
- **Arrange-Act-Assert**: Clear test structure
- **One Assertion Focus**: Test one thing per test
- **Descriptive Names**: Test name explains what's tested
- **Fast Unit Tests**: Keep under 1ms when possible

### 3. Mock Strategy
- **Mock at Boundaries**: External services, not internal logic
- **Verify Interactions**: Check mocks were called correctly
- **Realistic Responses**: Mock responses match real ones
- **Don't Over-Mock**: Too many mocks hide real bugs

This testing architecture ensures comprehensive coverage while maintaining fast feedback loops during development.