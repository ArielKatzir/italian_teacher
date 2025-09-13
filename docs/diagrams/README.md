# Architecture Diagrams

This directory contains the visual diagrams for the Italian Teacher multi-agent system architecture.

## Diagrams

### 1. System Architecture (`system_architecture.png`)
- **Size**: 550KB, 4800x3600px
- **Content**: Complete system overview showing all major components, their relationships, and data flow
- **Key Elements**: User interface, coordinator agent, specialized agents, infrastructure components, storage layer, and external services

### 2. Agent Selection Flow (`agent_selection_flow.png`)
- **Size**: 276KB, 4200x3000px
- **Content**: Decision flow for intelligent agent selection and context switching
- **Key Elements**: Message analysis, context switch detection, discovery service queries, agent scoring, and selection logic

### 3. Session Lifecycle (`session_lifecycle.png`)
- **Size**: 285KB, 4200x3000px
- **Content**: State machine showing session progression from creation to completion
- **Key Elements**: Session states, transitions, decision points, and method calls

### 4. Communication Patterns (`communication_patterns.png`)
- **Size**: 453KB, 4800x3600px (2x2 grid)
- **Content**: Four main interaction patterns in the system
- **Key Elements**: 
  - Direct agent communication
  - Agent collaboration via event bus
  - Context switching between agents
  - Load balancing across agent instances

### 5. Data Flow Diagram (`data_flow_diagram.png`)
- **Size**: 272KB, 4200x3000px
- **Content**: Data lifecycle management and retention policies
- **Key Elements**: Data stages, retention timeline, privacy compliance, and analytics flow

## Generation

All diagrams are generated programmatically using the script:
```bash
python scripts/generate_diagrams.py
```

## Technical Details

- **Format**: PNG with high resolution (300 DPI)
- **Library**: matplotlib with custom styling
- **Colors**: Consistent color palette across all diagrams
- **Fonts**: Arial for readability
- **Style**: Professional business diagram appearance

## Usage

These diagrams are referenced in the main architecture documentation:
- [`../core-architecture.md`](../core-architecture.md)

The diagrams can be embedded in presentations, documentation, or used for architectural reviews and discussions.