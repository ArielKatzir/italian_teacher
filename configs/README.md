# Configuration Structure

This directory contains the Italian Teacher configuration files organized by concern and environment.

## File Organization

### Core System Configuration
- **`system.yaml`** - Core system operational parameters (timeouts, thresholds, scoring weights)
- **`development.yaml`** - Development environment settings (app, database, models, agents)

### Agent Configurations  
- **`agents/`** - Individual agent personality and behavior configurations
  - `marco.yaml` - Friendly conversationalist agent
  - `professoressa_rossi.yaml` - Grammar expert agent
  - `nonna_giulia.yaml` - Cultural storyteller agent
  - `lorenzo.yaml` - Modern Italian specialist agent

## Configuration Hierarchy

### 1. System Configuration (`system.yaml`)
```yaml
system:
  agent_defaults:        # Default agent capabilities
  agent_limits:          # Operational limits
  registry:              # Agent registry settings
  discovery:             # Agent selection thresholds
    confidence_thresholds: # By user proficiency level
    complexity_thresholds: # By conversation complexity
  event_bus:             # Inter-agent communication
  retention:             # Data cleanup schedules
```

### 2. Application Configuration (`development.yaml`)
```yaml
app:                     # Application metadata
api:                     # API server settings
database:                # Database connection
redis:                   # Cache settings
models:                  # ML model configuration
agents:                  # Agent personality settings
coordinator:             # Agent coordination logic
training:                # ML training parameters
```

## Loading Priority

The configuration system searches for files in this order:

1. **Explicit path** (if provided via `init_config(path)`)
2. **`configs/system.yaml`** (primary system config location)
3. **`configs/config.yaml`** (alternative)
4. **`config.yaml`** (legacy root location)
5. **`~/.italian_teacher/system.yaml`** (user-specific)
6. **`~/.italian_teacher/config.yaml`** (legacy user)

## Environment Overrides

System settings can be overridden via environment variables:

```bash
# Agent configuration
export ITALIAN_TEACHER_MAX_CONCURRENT_SESSIONS=10
export ITALIAN_TEACHER_HEARTBEAT_TIMEOUT=600

# Discovery thresholds
export ITALIAN_TEACHER_CONFIDENCE_BEGINNER=0.5
export ITALIAN_TEACHER_MAX_CANDIDATES=5

# Event bus settings
export ITALIAN_TEACHER_EVENT_TIMEOUT=10.0
```

## Configuration Separation

### System vs Application Config

**System Config** (`system.yaml`):
- Operational parameters (timeouts, limits, thresholds)
- Agent registry and discovery settings
- Inter-agent communication parameters
- Data retention policies
- Performance tuning parameters

**Application Config** (`development.yaml`):
- Business logic settings (agent personalities)
- External service connections (database, Redis)
- Model configurations and training parameters
- API server settings

This separation allows:
- **System tuning** without touching business logic
- **Environment-specific** operational parameters
- **Independent scaling** of system vs application concerns
- **Clean deployment** configurations per environment

## Usage in Code

```python
from src.core.config import get_config

# Get system configuration
config = get_config()
timeout = config.get_registry_config().heartbeat_timeout
thresholds = config.get_discovery_config().confidence_thresholds

# Environment-specific loading
from src.core.config import init_config
config = init_config("configs/production.yaml")
```

## Adding New Configuration

### System Parameters
Add to `system.yaml` under appropriate section:
```yaml
system:
  new_subsystem:
    new_parameter: 42
```

### Application Settings  
Add to `development.yaml` or environment-specific files:
```yaml
new_feature:
  enabled: true
  settings: {...}
```

The configuration system provides graceful fallbacks - if any config file is missing or malformed, the system will use hardcoded defaults and continue operating normally.