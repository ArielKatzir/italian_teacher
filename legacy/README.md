# Legacy Code

This directory contains code that was part of the original multi-agent Italian Teacher system but is not currently needed for the working product (teacher homework assignment system).

## What's Here

### src/agents/
- **marco_agent.py** - Original Marco agent with personality system
- **base_agent.py** - Base agent framework
- Agent registry and discovery system
- Full multi-agent coordination infrastructure

### src/core/
- **agent_config.py** - Agent personality configuration
- **agent_events.py** - Event bus for agent communication
- **coordinator.py** - Multi-agent coordination
- **interfaces/** - Agent communication interfaces

### src/learning_units/
- Learning unit structure and progression
- Topic-based learning paths

### src/curriculum.py, service.py, validator.py
- Curriculum management system
- Educational service layer
- Question validation system

### src/prompts/
- Prompt templates for various educational scenarios

### src/inference/
- Old inference utilities (now in fine_tuning/)

### tests/unit/
- Tests for agent system components
- Tests for core infrastructure
- Tests for coordination and events

## Why It's Here

The project pivoted from a complex multi-agent system to focus on shipping a working product:
- **Original vision**: Multiple AI agents (Marco, Sofia, etc.) coordinating to teach Italian
- **Current focus**: Teacher homework assignment system with manual input
- **Decision**: Move unused complexity to legacy, ship working product first

## When to Use

Restore code from legacy/ when:
1. Adding multi-agent coordination back
2. Implementing AI-based command parsing (if needed)
3. Building student-facing agent interactions
4. Adding curriculum/learning path features

## What Was Kept

Currently active code (not in legacy/):
- `src/educational/teacher/` - Teacher homework assignment system
- `src/fine_tuning/` - Model training and inference
- `src/models/` - Model configurations
- `tests/unit/test_homework_assignment.py` - Assignment tests
- `tests/integration/test_teacher_student_flow.py` - Flow tests

## Moved On

**Date**: October 1, 2025
**Reason**: Simplify for product development, reduce complexity
**Tests**: All current tests still passing (18 unit + 9 integration = 27 total)

## Future

When we're ready to add back multi-agent features:
1. Review this legacy code
2. Update for current architecture
3. Test thoroughly
4. Integrate incrementally

For now, focus on shipping the homework system! 🚀
