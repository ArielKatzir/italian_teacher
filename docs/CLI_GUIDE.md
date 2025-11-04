# Italian Teacher CLI Guide

Command-line interface for teachers and students to manage homework assignments.

## 🚀 Quick Start

### Prerequisites

1. **Start the API**:
   ```bash
   ./run_api.sh
   ```

2. **(Optional) Start Colab GPU** for real exercise generation:
   - Open `demos/colab_inference_api.ipynb` in Google Colab
   - Enable GPU and run all cells
   - Copy the ngrok URL
   - Export it: `export INFERENCE_API_URL="https://your-url.ngrok-free.dev"`

### Install CLI Dependencies

```bash
pip install typer rich httpx
```

## 👨‍🏫 Teacher CLI

The teacher CLI allows you to manage students and create homework assignments.

### Student Management

#### Create a Student
```bash
./teacher student create --name "Mario Rossi" --email "mario@example.com"

# Or using Python directly:
python -m src.cli.teacher_cli student create --name "Mario Rossi" --email "mario@example.com"
```

**Output:**
```
✓ Student created successfully!
ID: 1
Name: Mario Rossi
Email: mario@example.com
```

#### List All Students
```bash
./teacher student list
```

**Output:**
```
Students
┏━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ ID     ┃ Name        ┃ Email             ┃ Created At          ┃
┡━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ 1      │ Mario Rossi │ mario@example.com │ 2025-10-03T09:48:10 │
└────────┴─────────────┴───────────────────┴─────────────────────┘

Total students: 1
```

#### Delete a Student
```bash
./teacher student delete --id 1

# You will be prompted for confirmation
```

**Note:** Deleting a student also deletes all their homework.

---

### Assignment Management

#### Create an Assignment

**Basic Usage:**
```bash
./teacher assignment create \
  --student-ids 1,2,3 \
  --level A2 \
  --topic "daily routines" \
  --grammar "present_tense" \
  --quantity 5
```

**All Options:**
```bash
./teacher assignment create \
  --student-ids 1,2,3 \              # Comma-separated student IDs (required)
  --level A2 \                       # CEFR level: A1, A2, B1, B2, C1, C2 (required)
  --topic "daily routines" \         # Topic (optional)
  --grammar "present_tense" \        # Grammar focus (optional)
  --quantity 5 \                     # Number of exercises (default: 5)
  --types "fill_in_blank,translation,multiple_choice"  # Exercise types (optional)
```

**Output:**
```
✓ Assignment created successfully!
Assignment ID: 1
Status: pending
CEFR Level: A2
Topic: daily routines
Grammar: present_tense
Quantity: 5
Students: 1, 2, 3

⚙ Background generation started. Use 'assignment status' to check progress.
```

**What happens?**
1. Assignment is created with status `pending`
2. Homework records are created for each student
3. Background task starts generating exercises using Colab GPU (if configured)
4. Status changes: `pending` → `generating` → `completed`

#### Check Assignment Status

```bash
./teacher assignment status --id 1
```

**Output (Pending):**
```
Assignment #1 Status
⏳ Status: PENDING
CEFR Level: A2
Topic: daily routines
Grammar: present_tense
Quantity: 5
Students: 1, 2, 3
Created: 2025-10-03T09:48:29

⏳ Assignment is queued for generation.
```

**Output (Generating):**
```
Assignment #1 Status
⚙️ Status: GENERATING
...
⚙️ Exercises are being generated using Colab GPU... Please wait.
```

**Output (Completed):**
```
Assignment #1 Status
✅ Status: COMPLETED
CEFR Level: A2
Topic: daily routines
Grammar: present_tense
Quantity: 5
Students: 1, 2, 3
Created: 2025-10-03T09:48:29
Completed: 2025-10-03T09:48:31

✅ Assignment completed! Students can now view their homework.
```

#### List All Assignments

```bash
./teacher assignment list
```

**Output:**
```
Assignments
┏━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ ID     ┃ Level    ┃ Topic        ┃ Students     ┃ Status       ┃ Created At  ┃
┡━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 1      │ A2       │ daily        │ 1, 2, 3      │ completed    │ 2025-10-03… │
│        │          │ routines     │              │              │             │
└────────┴──────────┴──────────────┴──────────────┴──────────────┴─────────────┘

Total assignments: 1
```

---

## 👨‍🎓 Student CLI

The student CLI allows students to view their assigned homework.

### Important: Homework IDs

**⚠️ Each homework record has a unique homework ID** (not the same as assignment ID).

- When an assignment is created for multiple students, each student gets their own homework record with a unique ID
- Example: Assignment #1 for students 1 and 2 creates:
  - Homework ID **1** for Student 1 (Assignment 1)
  - Homework ID **2** for Student 2 (Assignment 1)

**Always run `homework list` first to see which homework ID belongs to your student.**

### View Homework List

```bash
./student homework list --student-id 1

# Filter by status (default: available)
./student homework list --student-id 1 --status available
./student homework list --student-id 1 --status pending
./student homework list --student-id 1 --status completed
```

**Output:**
```
Homework for Student 1 (Status: available)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Homework ID  ┃ Assignment ID  ┃ # Exercises  ┃ Status       ┃ Created At     ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ 1            │ 1              │ 5            │ available    │ 2025-10-03T09… │
└──────────────┴────────────────┴──────────────┴──────────────┴────────────────┘

Total homework: 1

💡 To view exercises, use:
   homework view --student-id 1 --homework-id 1
```

**Note:** The homework ID in the first column is what you use with the `view` command.

### View Homework Details

```bash
# Use the homework ID from the list command above
./student homework view --student-id 1 --homework-id 1
```

**Common mistake:**
```bash
# ❌ WRONG - Using the same homework ID for different students
./student homework view --student-id 2 --homework-id 1
# Error: Homework 1 not found for student 2

# ✅ CORRECT - First list to find the right homework ID
./student homework list --student-id 2
# Then use the homework ID shown in the list (probably ID 2)
./student homework view --student-id 2 --homework-id 2
```

**Output:**
```
✅ Homework #1
Assignment ID: 1
Status: available
Created: 2025-10-03T09:48:29

Total Exercises: 5

╭────── Exercise 1: Fill In Blank ───────╮
│ Question: Io ___ a Roma ieri. (andare) │
│                                        │
│ Answer: sono andato                    │
│                                        │
│ Past tense of 'andare'                 │
╰────────────────────────────────────────╯

╭──────────── Exercise 2: Translation ─────────────╮
│ Question: Translate: I went to Milan yesterday.  │
│                                                  │
│ Answer: Sono andato a Milano ieri.               │
│                                                  │
│ CEFR A2 translation exercise                     │
╰──────────────────────────────────────────────────╯

... (3 more exercises)

Good luck with your homework! 🇮🇹
```

---

## 🔧 Configuration

### API Base URL

By default, the CLI connects to `http://localhost:8000`. To change this:

```bash
export API_BASE_URL="http://your-api-url:8000"
```

### Exercise Generation Modes

The system supports two modes:

#### 1. Mock Mode (Default)
- **Setup**: None required
- **Speed**: Instant (2 seconds)
- **Quality**: Simple placeholder exercises
- **Use case**: Testing, development

#### 2. Colab GPU Mode (Recommended)
- **Setup**: Start Colab inference service
- **Speed**: ~90-100 seconds for 5 exercises
- **Quality**: 100/100 quality score, authentic Italian exercises
- **Use case**: Production, real teaching

**Enable Colab GPU Mode:**
```bash
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"
```

See [docs/COLAB_GPU_SETUP.md](COLAB_GPU_SETUP.md) for detailed setup.

---

## 📋 Common Workflows

### Workflow 1: Teacher Creates Homework

1. **Create students**:
   ```bash
   ./teacher student create --name "Mario" --email "mario@example.com"
   ./teacher student create --name "Luigi" --email "luigi@example.com"
   ```

2. **Create assignment**:
   ```bash
   ./teacher assignment create \
     --student-ids 1,2 \
     --level A2 \
     --topic "daily routines" \
     --quantity 5
   ```

3. **Check status**:
   ```bash
   ./teacher assignment status --id 1
   ```

4. **Wait for completion** (or check periodically)

### Workflow 2: Student Views Homework

1. **List homework**:
   ```bash
   ./student homework list --student-id 1
   ```

2. **View specific homework**:
   ```bash
   ./student homework view --student-id 1 --homework-id 1
   ```

3. **Complete exercises** (on paper or in a separate system)

### Workflow 3: Teacher Manages Multiple Students

```bash
# Create multiple students
./teacher student create --name "Mario" --email "mario@example.com"
./teacher student create --name "Luigi" --email "luigi@example.com"
./teacher student create --name "Giulia" --email "giulia@example.com"

# View all students
./teacher student list

# Create assignment for specific students
./teacher assignment create \
  --student-ids 1,3 \
  --level B1 \
  --topic "Italian culture" \
  --grammar "past_tense" \
  --quantity 7

# View all assignments
./teacher assignment list

# Check specific assignment
./teacher assignment status --id 1
```

---

## ⚡ Tips & Tricks

### Shortcut Scripts

The helper scripts `./teacher` and `./student` are shortcuts for:
- `./teacher` = `python -m src.cli.teacher_cli`
- `./student` = `python -m src.cli.student_cli`

### Get Help

```bash
# General help
./teacher --help
./student --help

# Command-specific help
./teacher student --help
./teacher assignment --help
./student homework --help

# Subcommand help
./teacher student create --help
./teacher assignment create --help
```

### Check API Status

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{"status": "healthy"}
```

### View Exercises Without Student ID

If you're a teacher and want to preview exercises:

```bash
# List all assignments
./teacher assignment list

# Get assignment details (includes student IDs)
./teacher assignment status --id 1

# Then view as student
./student homework list --student-id <student_id>
```

---

## 🐛 Troubleshooting

### "Connection error" or "Connection refused"

**Problem:** API is not running

**Solution:**
```bash
# Start the API
./run_api.sh

# Or manually
python -m uvicorn src.api.main:app --port 8000
```

### "Student not found"

**Problem:** Invalid student ID

**Solution:**
```bash
# List all students to find valid IDs
./teacher student list
```

### "Assignment not found"

**Problem:** Invalid assignment ID

**Solution:**
```bash
# List all assignments to find valid IDs
./teacher assignment list
```

### Exercises are mock/placeholder

**Problem:** Colab GPU not configured

**Solution:**
1. Start Colab inference service (see [COLAB_GPU_SETUP.md](COLAB_GPU_SETUP.md))
2. Export ngrok URL:
   ```bash
   export INFERENCE_API_URL="https://your-url.ngrok-free.dev"
   ```
3. Restart API:
   ```bash
   ./run_api.sh
   ```

### Assignment stuck in "generating" status

**Problem:** Background task failed or Colab disconnected

**Solution:**
1. Check Colab notebook is still running
2. Verify ngrok URL is correct:
   ```bash
   curl $INFERENCE_API_URL/health
   ```
3. Check API logs for errors
4. If needed, delete and recreate the assignment

---

## 📚 Additional Resources

- **[QUICKSTART.md](../QUICKSTART.md)** - Get started in 3 minutes
- **[COLAB_GPU_SETUP.md](COLAB_GPU_SETUP.md)** - Complete Colab setup guide
- **[API_DEMO_GUIDE.md](../demos/API_DEMO_GUIDE.md)** - API usage examples
- **API Documentation** - Visit http://localhost:8000/docs

---

## 🎨 Example Session

```bash
# 1. Start API
./run_api.sh

# 2. Teacher: Create students
./teacher student create --name "Mario" --email "mario@example.com"
./teacher student create --name "Luigi" --email "luigi@example.com"

# 3. Teacher: Create assignment
./teacher assignment create \
  --student-ids 1,2 \
  --level A2 \
  --topic "daily routines" \
  --quantity 5

# Output: Assignment ID: 1

# 4. Teacher: Check status (wait a few seconds for mock generation)
./teacher assignment status --id 1

# 5. Student: View homework
./student homework list --student-id 1
./student homework view --student-id 1 --homework-id 1

# 6. Teacher: View all assignments
./teacher assignment list
```

---

**Built with ❤️ for Italian language learners**
