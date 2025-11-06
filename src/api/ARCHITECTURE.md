# Italian Teacher API - Architecture Documentation

This document provides a comprehensive overview of the Italian Teacher API architecture, data flows, and system components.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Component Details](#component-details)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Database Schema](#database-schema)
6. [API Endpoints](#api-endpoints)
7. [Sequence Diagrams](#sequence-diagrams)

---

## System Overview

The Italian Teacher API is a **dual-service architecture** that combines a local FastAPI server with GPU-accelerated inference running on Google Colab. The system enables teachers to create personalized Italian language homework assignments that are automatically generated using a fine-tuned GRPO (Group Relative Policy Optimization) language model.

### Key Features

- **Teacher Interface**: Create students, assign homework with custom parameters
- **Student Interface**: Retrieve personalized homework exercises
- **GPU Inference**: High-quality exercise generation using GRPO-trained model
- **Asynchronous Processing**: Background task processing with status tracking
- **Persistent Storage**: SQLite database for all records

---

## Architecture Diagram

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐       │
│  │ Teacher CLI │    │ Student CLI  │    │ Web Browser         │       │
│  │ (Terminal)  │    │ (Terminal)   │    │ (Swagger UI/ReDoc)  │       │
│  └──────┬──────┘    └──────┬───────┘    └──────────┬──────────┘       │
│         │                   │                       │                   │
└─────────┼───────────────────┼───────────────────────┼───────────────────┘
          │                   │                       │
          └───────────────────┴───────────────────────┘
                              │
                              ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────────────────┐
│                    SERVICE 1: MAIN API SERVER                            │
│                    (Local Machine - Port 8000)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────┐     │
│  │ FastAPI Application (main.py)                                 │     │
│  │  - CORS Middleware                                             │     │
│  │  - Lifespan Management                                         │     │
│  │  - Route Registration                                          │     │
│  └────────┬──────────────────────────────────────────┬───────────┘     │
│           │                                           │                  │
│           ↓                                           ↓                  │
│  ┌────────────────────┐                   ┌───────────────────┐        │
│  │ Teacher Routes     │                   │ Student Routes    │        │
│  │ (/api/teacher)     │                   │ (/api/student)    │        │
│  ├────────────────────┤                   ├───────────────────┤        │
│  │ • POST /students   │                   │ • GET /{id}/      │        │
│  │ • GET /students    │                   │   homework        │        │
│  │ • DELETE /students │                   │ • GET /{id}/      │        │
│  │ • POST /assignments│                   │   homework/{hw_id}│        │
│  │ • GET /assignments │                   └─────────┬─────────┘        │
│  │ • GET /assignments │                             │                  │
│  │   /{id}            │                             │                  │
│  └────────┬───────────┘                             │                  │
│           │                                          │                  │
│           └──────────────────┬───────────────────────┘                  │
│                              ↓                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Pydantic Schemas (schemas.py)                               │       │
│  │ - Request/Response Validation                               │       │
│  │ - Type Safety                                               │       │
│  └────────────────────────────┬────────────────────────────────┘       │
│                               ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Database Layer (database.py)                                │       │
│  │ - SQLAlchemy ORM Models                                     │       │
│  │ - Async Session Management                                  │       │
│  │ - Connection Pooling                                        │       │
│  └────────────────────────────┬────────────────────────────────┘       │
│                               ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ SQLite Database (data/italian_teacher.db)                   │       │
│  │ Tables: students, assignments, homework, assignment_students│       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Background Services (services/homework_service.py)          │       │
│  │ - Homework Generation Orchestration                         │       │
│  │ - Status Management                                         │       │
│  │ - Error Handling                                            │       │
│  └────────────────────────────┬────────────────────────────────┘       │
│                               │                                          │
└───────────────────────────────┼──────────────────────────────────────────┘
                                │ HTTP POST
                                │ /generate
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              SERVICE 2: GPU INFERENCE API                                │
│              (Google Colab - Port 8001 via ngrok)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ FastAPI Inference Service (inference/colab_api.py)          │       │
│  │ - POST /generate                                            │       │
│  │ - GET /health                                               │       │
│  │ - GET /                                                     │       │
│  └────────────────────────────┬────────────────────────────────┘       │
│                               ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ vLLM Inference Engine                                       │       │
│  │ - High-performance inference                                │       │
│  │ - GPU-accelerated generation                                │       │
│  │ - Continuous batching                                       │       │
│  │ - Optimized memory management                               │       │
│  └────────────────────────────┬────────────────────────────────┘       │
│                               ↓                                          │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ TeacherPet_italian_grpo Model                               │       │
│  │ - GRPO-trained (Reinforcement Learning)                     │       │
│  │ - Fine-tuned for Italian exercises                          │       │
│  │ - Grammar accuracy optimization                             │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ Post-Processing Pipeline                                    │       │
│  │ - JSON parsing (5 fallback strategies)                     │       │
│  │ - Grammar validation (spaCy)                               │       │
│  │ - Multiple choice validation                               │       │
│  │ - Tense consistency checking                               │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │ NVIDIA L4 GPU (Google Colab)                                │       │
│  │ - 24GB VRAM                                                 │       │
│  │ - 85% GPU utilization configured                            │       │
│  └─────────────────────────────────────────────────────────────┘       │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Service 1: Main API Server

#### 1. **main.py** - FastAPI Application Entry Point
- Initializes database on startup
- Configures CORS middleware for frontend integration
- Registers route handlers
- Provides health check and root endpoints

#### 2. **routes/** - API Endpoint Handlers
- **teacher.py**: Teacher-specific operations
  - Student management (CRUD)
  - Assignment creation and management
  - Background task triggering
- **student.py**: Student-specific operations
  - Homework retrieval with filtering
  - Individual homework access

#### 3. **schemas.py** - Pydantic Models
- Request validation schemas
- Response serialization schemas
- Type safety enforcement
- Automatic API documentation

#### 4. **database.py** - Data Layer
- SQLAlchemy ORM models
- Async session management
- Database initialization
- Relationship definitions

#### 5. **services/homework_service.py** - Business Logic
- Orchestrates homework generation
- Manages assignment lifecycle
- Handles errors and retries
- Communicates with GPU inference service

### Service 2: GPU Inference API

#### 1. **inference/colab_api.py** - Inference Service
- FastAPI application factory
- vLLM integration
- Request/response handling
- Multi-strategy parsing
- Grammar validation

---

## Data Flow Diagrams

### Flow 1: Creating a Homework Assignment

```
┌──────────┐
│ Teacher  │
│ (Client) │
└─────┬────┘
      │
      │ 1. POST /api/teacher/assignments
      │    {
      │      cefr_level: "B1",
      │      grammar_focus: "past_tense",
      │      topic: "Italian art",
      │      quantity: 5,
      │      exercise_types: ["fill_in_blank", "multiple_choice"],
      │      student_ids: [1, 2, 3]
      │    }
      ↓
┌─────────────────────────────────────────────┐
│ Main API Server - Teacher Routes           │
│                                             │
│  2. Validate request (Pydantic)             │
│  3. Check students exist in database        │
│  4. Create Assignment record (status: pending)│
│  5. Create Homework records for each student│
│     (status: pending)                       │
│  6. Create AssignmentStudent junction records│
│  7. Commit to database                      │
│  8. Add background task                     │
│  9. Return assignment response immediately  │
└─────────────┬───────────────────────────────┘
              │
              │ 10. Response to client
              │     {
              │       id: 123,
              │       status: "pending",
              │       student_ids: [1,2,3],
              │       ...
              │     }
              ↓
        ┌──────────┐
        │ Teacher  │
        └──────────┘

    [BACKGROUND PROCESSING STARTS]

┌─────────────────────────────────────────────┐
│ Homework Service (Background Task)          │
│                                             │
│  11. Update assignment status: "generating" │
│  12. For each student:                      │
│      ┌────────────────────────────────────┐ │
│      │ a. Call GPU inference service      │ │
│      │ b. Receive generated exercises     │ │
│      │ c. Update homework record          │ │
│      │ d. Set homework status: "available"│ │
│      └────────────────────────────────────┘ │
│  13. Update assignment status: "completed"  │
│  14. Set completed_at timestamp             │
└─────────────┬───────────────────────────────┘
              │
              │ HTTP POST to GPU service
              ↓
┌─────────────────────────────────────────────┐
│ GPU Inference Service                       │
│                                             │
│  15. Receive generation request             │
│  16. Construct optimized prompt             │
│  17. Call vLLM with sampling params         │
│  18. Generate exercises (JSON)              │
│  19. Parse with fallback strategies         │
│  20. Validate grammar with spaCy            │
│  21. Validate multiple choice options       │
│  22. Return exercise list                   │
└─────────────┬───────────────────────────────┘
              │
              │ Return exercises JSON
              ↓
┌─────────────────────────────────────────────┐
│ Homework Service                            │
│                                             │
│  23. Store exercises in homework.exercises_json│
│  24. Continue for remaining students        │
└─────────────────────────────────────────────┘
```

### Flow 2: Student Retrieving Homework

```
┌──────────┐
│ Student  │
│ (Client) │
└─────┬────┘
      │
      │ 1. GET /api/student/123/homework?status=available
      ↓
┌─────────────────────────────────────────────┐
│ Main API Server - Student Routes           │
│                                             │
│  2. Validate student_id                     │
│  3. Check student exists in database        │
│  4. Query homework with filters:            │
│     - student_id = 123                      │
│     - status = "available" (if specified)   │
│     - ORDER BY created_at DESC              │
│  5. Deserialize exercises_json              │
│  6. Format response                         │
└─────────────┬───────────────────────────────┘
              │
              │ 7. Return homework list
              │    {
              │      homework: [
              │        {
              │          id: 456,
              │          assignment_id: 123,
              │          exercises: [
              │            {
              │              type: "fill_in_blank",
              │              question: "Il Rinascimento ___ a Firenze.",
              │              correct_answer: "è iniziato",
              │              explanation: "Past tense usage..."
              │            },
              │            ...
              │          ],
              │          status: "available",
              │          created_at: "2025-11-04T10:00:00"
              │        }
              │      ],
              │      total: 1
              │    }
              ↓
        ┌──────────┐
        │ Student  │
        └──────────┘
```

### Flow 3: Monitoring Assignment Status

```
┌──────────┐
│ Teacher  │
└─────┬────┘
      │
      │ 1. GET /api/teacher/assignments/123
      ↓
┌─────────────────────────────────────────────┐
│ Main API Server - Teacher Routes           │
│                                             │
│  2. Query assignment by ID                  │
│  3. Query student IDs from junction table   │
│  4. Format response with current status     │
└─────────────┬───────────────────────────────┘
              │
              │ 5. Return assignment with status
              │    {
              │      id: 123,
              │      status: "completed",  // or "pending", "generating", "failed"
              │      student_ids: [1, 2, 3],
              │      completed_at: "2025-11-04T10:05:00",
              │      error_message: null
              │    }
              ↓
        ┌──────────┐
        │ Teacher  │
        └──────────┘
```

---

## Database Schema

### Entity-Relationship Diagram

```
┌─────────────────────────┐
│ Student                 │
├─────────────────────────┤
│ PK  id (INTEGER)        │
│     name (STRING)       │
│ UQ  email (STRING)      │
│     created_at (DATETIME)│
└────────┬────────────────┘
         │
         │ 1:N
         │
         ↓
┌─────────────────────────┐       ┌─────────────────────────┐
│ Homework                │   N:1 │ Assignment              │
├─────────────────────────┤◄──────├─────────────────────────┤
│ PK  id (INTEGER)        │       │ PK  id (INTEGER)        │
│ FK  assignment_id       │       │     cefr_level (STRING) │
│ FK  student_id          │       │     grammar_focus (STR) │
│     exercises_json (JSON)│      │     topic (STRING)      │
│     status (STRING)     │       │     quantity (INTEGER)  │
│     created_at (DATETIME)│      │     exercise_types (JSON)│
│     completed_at (DATETIME)│    │     status (STRING)     │
└─────────────────────────┘       │     created_at (DATETIME)│
                                  │     completed_at (DATETIME)│
                                  │     error_message (TEXT)│
                                  └────────┬────────────────┘
                                           │
                                           │ 1:N
                                           │
                                           ↓
                                  ┌─────────────────────────┐
                                  │ AssignmentStudent       │
                                  │ (Junction Table)        │
                                  ├─────────────────────────┤
                                  │ PK  id (INTEGER)        │
                                  │ FK  assignment_id       │
                                  │ FK  student_id          │
                                  │     created_at (DATETIME)│
                                  └─────────────────────────┘
```

### Table Descriptions

#### **students**
- Stores student information
- Unique email constraint
- Cascading deletes to homework and assignment_students

#### **assignments**
- Teacher's homework specification
- Contains exercise generation parameters
- Status tracking: pending → generating → completed/failed
- Error logging for failed generations

#### **homework**
- Individual homework per student per assignment
- exercises_json: Array of exercise objects
- Status tracking: pending → available → in_progress → completed
- Links student to assignment

#### **assignment_students**
- Junction table for many-to-many relationship
- Tracks which students are assigned to which homework
- Enables assignment-level queries

### Status Lifecycle

```
Assignment Status Flow:
pending → generating → completed
                   ↓
                 failed (with error_message)

Homework Status Flow:
pending → available → in_progress → completed
```

---

## API Endpoints

### Service 1: Main API (Port 8000)

#### Root Endpoints
- `GET /` - API information and endpoint listing
- `GET /health` - Health check

#### Teacher Endpoints (`/api/teacher`)
- `POST /students` - Create new student
- `GET /students` - List all students
- `DELETE /students/{student_id}` - Delete student
- `POST /assignments` - Create homework assignment
- `GET /assignments` - List all assignments
- `GET /assignments/{assignment_id}` - Get assignment details

#### Student Endpoints (`/api/student`)
- `GET /{student_id}/homework` - Get student's homework list
- `GET /{student_id}/homework/{homework_id}` - Get specific homework

### Service 2: GPU Inference API (Port 8001)

- `GET /` - Service information
- `GET /health` - Detailed GPU health check
- `POST /generate` - Generate Italian exercises

---

## Sequence Diagrams

### Sequence 1: Complete Assignment Creation Flow

```
Teacher    Main API     Database    BG Service    GPU API    vLLM Model
   │           │            │            │            │           │
   │─POST─────>│            │            │            │           │
   │ /assignments           │            │            │           │
   │           │            │            │            │           │
   │           │──Validate──>            │            │           │
   │           │<─Students──│            │            │           │
   │           │    exist                │            │           │
   │           │            │            │            │           │
   │           │──INSERT────>            │            │           │
   │           │  Assignment             │            │           │
   │           │<─Created───│            │            │           │
   │           │            │            │            │           │
   │           │──INSERT────>            │            │           │
   │           │  Homework (x3)          │            │           │
   │           │<─Created───│            │            │           │
   │           │            │            │            │           │
   │<─Response─│            │            │            │           │
   │ {id:123}               │            │            │           │
   │ status:pending         │            │            │           │
   │           │            │            │            │           │
   │           │─Trigger────>            │            │           │
   │           │ BG Task                 │            │           │
   │           │            │            │            │           │
   │           │            │            │─UPDATE────>            │
   │           │            │            │ status:generating      │
   │           │            │            │            │           │
   │           │            │            │─POST──────>            │
   │           │            │            │ /generate              │
   │           │            │            │            │           │
   │           │            │            │            │─Prompt───>│
   │           │            │            │            │           │
   │           │            │            │            │<─Generate─│
   │           │            │            │            │ exercises │
   │           │            │            │            │           │
   │           │            │            │<─JSON─────│            │
   │           │            │            │ exercises               │
   │           │            │            │            │           │
   │           │            │            │─UPDATE────>            │
   │           │            │            │ homework.exercises_json│
   │           │            │            │            │           │
   │           │            │            │ [Repeat for each student]
   │           │            │            │            │           │
   │           │            │            │─UPDATE────>            │
   │           │            │            │ status:completed       │
   │           │            │            │            │           │
```

### Sequence 2: Student Homework Retrieval

```
Student    Main API     Database
   │           │            │
   │─GET──────>│            │
   │ /student/123/homework  │
   │           │            │
   │           │──Query────>│
   │           │ Student exists
   │           │<─Exists───│
   │           │            │
   │           │──Query────>│
   │           │ Homework    │
   │           │ WHERE student_id=123
   │           │ AND status='available'
   │           │<─Homework──│
   │           │   records   │
   │           │            │
   │           │─Deserialize│
   │           │ exercises_json
   │           │            │
   │<─Response─│            │
   │ {homework:[...]}       │
```

---

## Key Design Decisions

### 1. **Dual-Service Architecture**
- **Rationale**: Separates resource-intensive GPU inference from lightweight API operations
- **Benefits**: Scalability, cost optimization (GPU only when needed), flexibility

### 2. **Asynchronous Background Processing**
- **Rationale**: Exercise generation can take 30-60 seconds per student
- **Benefits**: Non-blocking API responses, better user experience, status monitoring

### 3. **Multi-Strategy JSON Parsing**
- **Rationale**: LLM output can be inconsistent or malformed
- **Benefits**: Robust parsing, 5 fallback strategies ensure reliability

### 4. **GRPO Model Training**
- **Rationale**: Reinforcement learning improves grammar accuracy
- **Benefits**: Better exercise quality, tense consistency, reduced hallucination

### 5. **SQLite for Local Development**
- **Rationale**: Simple setup, no external dependencies
- **Production**: Can easily swap to PostgreSQL via SQLAlchemy

### 6. **Pydantic Schemas**
- **Rationale**: Type safety and automatic validation
- **Benefits**: Better API docs, fewer bugs, clear contracts

---

## Performance Characteristics

### Main API (Service 1)
- **Startup Time**: ~1-2 seconds
- **Response Time**:
  - GET endpoints: 10-50ms
  - POST endpoints: 50-200ms (excluding background tasks)
- **Throughput**: Handles 100+ req/s on modest hardware

### GPU Inference API (Service 2)
- **Startup Time**: ~30 seconds (model loading)
- **Inference Time**:
  - 5 exercises: 5-10 seconds
  - 10 exercises: 15-25 seconds
- **Throughput**: Sequential processing (1 request at a time)
- **GPU Memory**: 8-12GB VRAM used

---

## Security Considerations

### Current Implementation (Development)
- CORS: Allow all origins (*)
- No authentication/authorization
- SQLite database with no encryption
- ngrok tunnel for public access

### Production Recommendations
1. **Authentication**: Add JWT or OAuth2
2. **CORS**: Restrict to specific origins
3. **Database**: Migrate to PostgreSQL with SSL
4. **Rate Limiting**: Implement per-user rate limits
5. **Input Validation**: Already using Pydantic (✓)
6. **HTTPS**: Required for production
7. **API Keys**: Protect inference endpoint

---

## Deployment Architecture

### Development (Current)
```
Local Machine (Mac)          Google Colab (GPU)
├── Main API (8000)          ├── Inference API (8001)
├── SQLite DB                ├── vLLM Engine
└── Python 3.9+              ├── GRPO Model
                             └── ngrok Tunnel
```

### Production (Recommended)
```
Cloud Provider (e.g., AWS/GCP)
├── Application Tier
│   ├── Main API (ECS/K8s)
│   ├── Load Balancer
│   └── Auto-scaling
├── Database Tier
│   ├── PostgreSQL (RDS)
│   └── Connection Pooling
└── GPU Tier
    ├── Inference Service (Dedicated GPU instances)
    ├── vLLM Engine
    └── Model Cache
```

---

## Error Handling

### Assignment Generation Failures
1. **GPU Service Unavailable**: Assignment status set to "failed" with error message
2. **Model Generation Error**: Retries not implemented, marked as failed
3. **JSON Parsing Failure**: Multiple fallback strategies attempt recovery
4. **Network Timeout**: 180-second timeout on GPU requests

### Database Errors
- SQLAlchemy automatic rollback on exceptions
- Connection pool handles connection failures
- Cascade deletes prevent orphaned records

---

## Monitoring and Observability

### Current Logging
- Console logging for both services
- SQLAlchemy query logging (echo=True)
- Status tracking in database
- Error messages stored in database

### Production Recommendations
1. **Structured Logging**: JSON logs
2. **Metrics**: Prometheus + Grafana
3. **Tracing**: OpenTelemetry
4. **Alerting**: PagerDuty/Slack
5. **Health Checks**: Kubernetes liveness/readiness probes

---

## Testing Strategy

### Unit Tests
- Pydantic schema validation
- Database model methods
- Parsing strategies

### Integration Tests
- API endpoint tests
- Database operations
- Background task execution

### End-to-End Tests
- Complete homework generation flow
- Student retrieval workflows

### Load Tests
- Concurrent assignment creation
- High-volume student queries

---

## Future Enhancements

### Short Term
1. Add student authentication
2. Implement homework submission and grading
3. Add teacher dashboard
4. Improve error recovery (retries)

### Medium Term
1. Real-time status updates (WebSockets)
2. Exercise preview before assignment
3. Bulk student import
4. Assignment templates

### Long Term
1. Multi-language support
2. Adaptive difficulty (based on student performance)
3. Gamification and progress tracking
4. Mobile app integration

---

## Conclusion

The Italian Teacher API demonstrates a modern, scalable architecture for AI-powered educational applications. The dual-service design efficiently leverages GPU resources while maintaining a responsive user experience through asynchronous processing and comprehensive status tracking.
