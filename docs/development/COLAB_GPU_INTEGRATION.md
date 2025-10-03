# Colab GPU Integration for Italian Teacher API

**Complete guide to using Google Colab GPU for homework generation from your local API**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Why Use Colab GPU?](#why-use-colab-gpu)
4. [Setup Guide](#setup-guide)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting](#troubleshooting)
7. [Cost Analysis](#cost-analysis)
8. [Production Considerations](#production-considerations)

---

## Overview

This integration allows you to run the Italian Teacher API locally on your Mac while leveraging Google Colab's GPU for expensive inference operations. Your local API handles database, student management, and business logic, while Colab provides GPU-accelerated exercise generation using the fine-tuned Marco v3 model.

**Key Benefits:**
- 🚀 **4.4x faster inference** using vLLM on GPU (88.2 vs 23.8 tokens/sec)
- 💰 **Free GPU access** via Colab (or $10/month for Colab Pro)
- 🔧 **No local GPU required** - works on any Mac
- 🎯 **Production-ready** architecture with fallback to mock mode

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER WORKFLOW                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               LOCAL MAC (Your Computer)                     │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │         FastAPI Server (main.py)                   │    │
│  │  Port: 8000                                        │    │
│  │                                                     │    │
│  │  • Teacher endpoints (create students/assignments) │    │
│  │  • Student endpoints (get homework)                │    │
│  │  • Business logic & validation                     │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │    SQLite Database (italian_teacher.db)            │    │
│  │                                                     │    │
│  │  • Students table                                  │    │
│  │  • Assignments table                               │    │
│  │  • Homework table (stores generated exercises)     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │    Homework Service (homework_service.py)          │    │
│  │                                                     │    │
│  │  async def generate_exercises():                   │    │
│  │    if INFERENCE_API_URL:                           │    │
│  │      → HTTP POST to Colab                         │◄───┼──── HTTP Request
│  │    else:                                           │    │     (via internet)
│  │      → Mock generation (fallback)                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP POST /generate
                              │ JSON: {cefr_level, grammar_focus, ...}
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    NGROK TUNNEL                             │
│  Public URL: https://abc123.ngrok.io                        │
│  • Exposes Colab to internet                                │
│  • HTTPS encryption                                         │
│  • Free tier: 2 hour sessions                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              GOOGLE COLAB (GPU Instance)                    │
│  GPU: T4 (free) or L4/A100 (Colab Pro)                      │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │   FastAPI Inference Service (port 8000)            │    │
│  │   File: demos/colab_inference_api.ipynb            │    │
│  │                                                     │    │
│  │   POST /generate                                   │    │
│  │   • Receives exercise parameters                   │    │
│  │   • Calls vLLM for inference                       │    │
│  │   • Returns JSON exercises                         │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │        vLLM Inference Engine                       │    │
│  │                                                     │    │
│  │  • FlashAttention optimization                     │    │
│  │  • KV cache optimization                           │    │
│  │  • FP16 precision                                  │    │
│  │  • 88.2 tokens/sec throughput                      │    │
│  └────────────────┬───────────────────────────────────┘    │
│                   │                                         │
│  ┌────────────────▼───────────────────────────────────┐    │
│  │   Marco v3 LoRA Model (in Google Drive)            │    │
│  │   Path: models/minerva_marco_v3_merged             │    │
│  │                                                     │    │
│  │  • Fine-tuned on 17,913 Italian teaching examples  │    │
│  │  • Zero German contamination                       │    │
│  │  • Professional pedagogical structure              │    │
│  │  • ~6GB GPU memory                                 │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

1. **User creates assignment** via local API (e.g., `POST /api/teacher/assignments`)
2. **Local API** validates request, creates DB records, triggers background task
3. **homework_service.py** calls `generate_exercises()`
4. **HTTP POST** sent to ngrok URL → forwarded to Colab
5. **Colab FastAPI** receives request, formats prompt for Marco v3
6. **vLLM** generates exercises using GPU (2-5 seconds)
7. **JSON response** returned to local API via ngrok
8. **Local API** stores exercises in SQLite database
9. **Students** can retrieve homework via `GET /api/student/{id}/homework`

---

## Why Use Colab GPU?

### Performance Comparison

| Metric | Local CPU (Mac) | Colab T4 GPU (Free) | Improvement |
|--------|----------------|---------------------|-------------|
| Tokens/sec | ~23.8 | ~88.2 | **4.4x faster** |
| Inference time | ~4.2s | ~0.95s | **4.4x faster** |
| Memory | N/A (limited) | 16GB VRAM | Unlimited |
| Cost | $0 | $0 (free tier) | **Free!** |

### Real-World Impact

**Scenario: Teacher assigns homework to 20 students**

- **Without GPU (mock mode):**
  - 20 × 2s delay = 40 seconds total
  - Simple mock exercises only

- **With Colab GPU:**
  - 20 × 0.95s = 19 seconds total
  - **Authentic Italian exercises from Marco v3**
  - Professional grammar explanations
  - CEFR-level appropriate content

### Cost Analysis

| Colab Tier | GPU Type | Cost | Sessions | Best For |
|------------|----------|------|----------|----------|
| Free | T4 | $0/month | ~90 min idle timeout | Development, testing |
| Colab Pro | T4/L4/A100 | $10/month | Longer sessions, priority | Production, heavy use |
| Colab Pro+ | A100 | $50/month | Background execution | 24/7 production |

**Recommendation:** Start with free tier, upgrade to Pro ($10/mo) if you need longer sessions.

---

## Setup Guide

### Prerequisites

1. **Google Account** with Google Drive access
2. **ngrok Account** (free tier: https://ngrok.com)
3. **Local Python environment** with API dependencies

### Step 1: Install ngrok

1. Create free account at https://ngrok.com
2. Get your auth token from https://dashboard.ngrok.com/get-started/your-authtoken
3. Save the token (you'll need it in Step 3)

### Step 2: Open Colab Notebook

1. **Navigate to notebook** in Google Drive:
   ```
   italian_teacher/demos/colab_inference_api.ipynb
   ```

2. **Open with Google Colab**:
   - Right-click → "Open with" → "Google Colaboratory"
   - Or visit: https://colab.research.google.com and open from Drive

3. **Enable GPU**:
   - Click "Runtime" → "Change runtime type"
   - Hardware accelerator: **GPU**
   - GPU type: **T4** (free) or **L4/A100** (Pro)
   - Click "Save"

### Step 3: Run Colab Notebook Cells

**Execute each cell in order:**

#### Cell 1: Install Dependencies
```python
!pip install fastapi uvicorn pyngrok vllm nest-asyncio -q
```

#### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

#### Cell 3: Setup Model Path
```python
PROJECT_ROOT = "/content/drive/MyDrive/Colab Notebooks/italian_teacher"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/minerva_marco_v3_merged")
```

#### Cell 4: Load Marco v3 with vLLM
```python
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="half",
    max_model_len=2048,
    gpu_memory_utilization=0.85
)
```
⏳ **This takes 2-3 minutes** - loading 7B model into GPU

#### Cell 5: Create FastAPI App
```python
# FastAPI app with /generate endpoint
# (Cell creates the inference service)
```

#### Cell 6: Setup ngrok Tunnel
```python
NGROK_AUTH_TOKEN = "YOUR_TOKEN_HERE"  # ← PASTE YOUR TOKEN HERE
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
public_url = ngrok.connect(8000)
```

**📍 IMPORTANT:** Copy the public URL from the output!
```
🌐 ngrok tunnel created!
📍 Public URL: https://abc123-def456.ngrok.io
```

#### Cell 7: Start FastAPI Server
```python
# Starts server in background thread
```

#### Cell 8: Test Locally
```python
# Tests the /generate endpoint
```

### Step 4: Configure Local API

1. **Copy ngrok URL** from Colab output (e.g., `https://abc123.ngrok.io`)

2. **Set environment variable** on your Mac:
   ```bash
   export INFERENCE_API_URL="https://abc123.ngrok.io"
   ```

3. **Start local API**:
   ```bash
   cd "/Users/arielkatzir/Library/CloudStorage/GoogleDrive-ari.katzir@gmail.com/My Drive/Colab Notebooks/italian_teacher"
   ./run_api.sh
   ```

4. **Verify integration**:
   ```bash
   # You should see:
   # 🔥 Starting API with Colab GPU integration
   # 📡 Inference URL: https://abc123.ngrok.io
   ```

### Step 5: Test End-to-End

1. **Open API docs**: http://localhost:8000/docs

2. **Create a student**:
   ```json
   POST /api/teacher/students
   {
     "name": "Test Student",
     "email": "test@example.com"
   }
   ```

3. **Create assignment** (triggers Colab GPU):
   ```json
   POST /api/teacher/assignments
   {
     "cefr_level": "A2",
     "grammar_focus": "present_tense",
     "topic": "daily routines",
     "quantity": 3,
     "exercise_types": ["fill_in_blank", "translation"],
     "student_ids": [1]
   }
   ```

4. **Check Colab logs** - you should see:
   ```
   ✅ Generated 3 exercises via Colab GPU (1.2s, 156 tokens)
   ```

5. **Retrieve homework**:
   ```json
   GET /api/student/1/homework?status=available
   ```

**✅ Success!** You're now generating authentic Italian exercises with Colab GPU!

---

## Usage Examples

### Example 1: Create Assignment with Colab GPU

```bash
curl -X POST "http://localhost:8000/api/teacher/assignments" \
  -H "Content-Type: application/json" \
  -d '{
    "cefr_level": "B1",
    "grammar_focus": "subjunctive",
    "topic": "Italian culture",
    "quantity": 5,
    "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
    "student_ids": [1, 2, 3]
  }'
```

**What happens:**
1. Local API creates assignment record (status: `pending`)
2. Background task calls `homework_service.generate_exercises()`
3. HTTP POST sent to Colab ngrok URL
4. Colab vLLM generates 5 exercises per student (15 total)
5. Exercises stored in SQLite database
6. Assignment status updated to `completed`

**Timeline:**
- Assignment creation: ~50ms
- GPU generation (15 exercises): ~4-6 seconds
- Total: **~6 seconds** for professional Italian exercises

### Example 2: Check Generation Status

```bash
# Immediately after creating assignment
curl "http://localhost:8000/api/teacher/assignments/1"

# Response:
{
  "id": 1,
  "status": "generating",  ← In progress
  "completed_at": null
}

# After 5-6 seconds
curl "http://localhost:8000/api/teacher/assignments/1"

# Response:
{
  "id": 1,
  "status": "completed",  ← Done!
  "completed_at": "2025-10-01T23:45:12.123456"
}
```

### Example 3: Student Views Homework

```bash
curl "http://localhost:8000/api/student/1/homework?status=available"
```

**Response:**
```json
{
  "homework": [
    {
      "id": 1,
      "exercises": [
        {
          "type": "fill_in_blank",
          "question": "Io ___ alla festa se avessi tempo. (andare - congiuntivo)",
          "correct_answer": "andrei",
          "explanation": "The subjunctive 'andrei' expresses a hypothetical action..."
        },
        {
          "type": "translation",
          "question": "Translate: I would go to the party if I had time.",
          "correct_answer": "Andrei alla festa se avessi tempo.",
          "explanation": "B1 level conditional + subjunctive construction..."
        }
      ],
      "status": "available"
    }
  ]
}
```

### Example 4: Fallback to Mock Mode

```bash
# Without INFERENCE_API_URL set
unset INFERENCE_API_URL
./run_api.sh

# Server logs show:
# ⚠️  Starting API in mock mode (no GPU)
# 💡 To use Colab GPU, set INFERENCE_API_URL

# Assignments still work, but use simple mock exercises
# Useful for development without Colab
```

---

## Troubleshooting

### Issue 1: Colab Disconnects After 90 Minutes

**Problem:** Free Colab disconnects after idle timeout (~90 min)

**Solutions:**
1. **Colab Pro** ($10/mo) - longer sessions, priority access
2. **Keep-alive script** - run in Colab:
   ```python
   import time
   while True:
       print(".", end="")
       time.sleep(60)
   ```
3. **Restart when needed** - Just re-run notebook cells

### Issue 2: ngrok Tunnel Expires

**Problem:** Free ngrok tunnels expire after 2 hours

**Solutions:**
1. **Re-run ngrok cell** in Colab notebook
2. **Update `INFERENCE_API_URL`** with new URL:
   ```bash
   export INFERENCE_API_URL="https://NEW_URL.ngrok.io"
   # Restart local API
   ./run_api.sh
   ```
3. **Upgrade to ngrok Pro** ($8/mo) - persistent URLs

### Issue 3: Slow First Request

**Problem:** First generation takes 10-15 seconds

**Cause:** Model warmup and KV cache initialization

**Solution:** This is normal - subsequent requests are fast (~1-2s)

### Issue 4: Out of GPU Memory

**Problem:** `CUDA out of memory` error in Colab

**Solutions:**
1. **Restart Colab runtime**:
   - Runtime → Factory reset runtime
   - Re-run all cells
2. **Reduce `max_model_len`** in vLLM config:
   ```python
   llm = LLM(model=MODEL_PATH, max_model_len=1024)  # Was 2048
   ```
3. **Use smaller batches** - reduce `quantity` in requests

### Issue 5: Colab Can't Find Model

**Problem:** `Model not found at models/minerva_marco_v3_merged`

**Solutions:**
1. **Check model path** in Colab cell:
   ```python
   !ls -la "/content/drive/MyDrive/Colab Notebooks/italian_teacher/models"
   ```
2. **Update `MODEL_PATH`** if different:
   ```python
   MODEL_PATH = "/content/drive/MyDrive/YOUR_PATH/models/minerva_marco_v3_merged"
   ```

### Issue 6: Local API Can't Reach Colab

**Problem:** `Connection refused` or timeout errors

**Checklist:**
1. ✅ Colab notebook still running?
2. ✅ ngrok tunnel active? (Check ngrok cell output)
3. ✅ `INFERENCE_API_URL` set correctly?
   ```bash
   echo $INFERENCE_API_URL  # Should show https://...ngrok.io
   ```
4. ✅ Firewall blocking requests? (Unlikely with ngrok)
5. ✅ Try health check:
   ```bash
   curl "https://your-ngrok-url.ngrok.io/health"
   ```

### Issue 7: Exercises Are Still Mock Data

**Problem:** Homework doesn't use real Marco v3 model

**Debug:**
1. **Check environment variable**:
   ```bash
   echo $INFERENCE_API_URL
   # Should output: https://abc123.ngrok.io
   ```

2. **Check server logs** for:
   ```
   ✅ Generated X exercises via Colab GPU
   # vs
   ⚠️  INFERENCE_API_URL not set. Using mock generation.
   ```

3. **Restart API** after setting variable:
   ```bash
   export INFERENCE_API_URL="https://your-url.ngrok.io"
   ./run_api.sh
   ```

---

## Cost Analysis

### Monthly Cost Comparison

| Scenario | Setup | Monthly Cost | Notes |
|----------|-------|--------------|-------|
| **Development** | Free Colab + Free ngrok | **$0** | Perfect for testing |
| **Light Production** | Colab Pro + Free ngrok | **$10** | ~100 assignments/day |
| **Heavy Production** | Colab Pro+ + ngrok Pro | **$58** | 24/7 uptime |
| **Cloud GPU** | RunPod/Lambda L4 GPU | **~$200** | Always-on alternative |

### Usage-Based Costs

**Free Tier (Recommended for Development):**
- ✅ Colab: Free
- ✅ ngrok: Free (2hr sessions, 1 tunnel)
- ✅ Google Drive: 15GB free (model ~6GB)
- **Total: $0/month**

**Production Tier (Recommended for Teachers):**
- 💰 Colab Pro: $10/month
  - Longer sessions (~24 hours)
  - Priority GPU access
  - Background execution (Pro+: $50/mo)
- 💰 ngrok Pro: $8/month (optional)
  - Persistent URLs (no expiry)
  - More tunnels
- **Total: $10-18/month**

### ROI Analysis

**Scenario: Italian language teacher with 30 students**

- **Manual homework creation:** 2 hours/week × $50/hr = **$400/month**
- **Colab Pro automation:** $10/month + 5 min/week = **$10/month**
- **Savings: $390/month (98% reduction)**

---

## Production Considerations

### Security

✅ **HTTPS Encryption:** ngrok provides automatic HTTPS
✅ **No Authentication Required:** Colab is private (not exposed directly)
⚠️ **ngrok URL Security:** ngrok URLs are obscure but public

**Recommendations:**
1. **API Key Authentication** (future enhancement):
   ```python
   @app.post("/generate")
   async def generate(request: Request, api_key: str = Header(...)):
       if api_key != os.getenv("API_KEY"):
           raise HTTPException(401)
   ```

2. **Rate Limiting** (future enhancement)

### Reliability

| Issue | Mitigation |
|-------|------------|
| **Colab disconnect** | Automatic fallback to mock mode |
| **ngrok expiry** | Monitor and restart tunnel |
| **GPU unavailable** | Queue requests, retry logic |

**Current Implementation:**
- ✅ Automatic fallback to mock if Colab unreachable
- ✅ 60-second timeout on HTTP requests
- ✅ Error logging for debugging

### Scalability

**Current Architecture:**
- ✅ Single GPU: ~50-100 requests/hour
- ✅ Perfect for individual teachers (1-30 students)
- ⚠️ Not suitable for school-wide deployment (100+ students)

**Scaling Options:**
1. **Multiple Colab instances** with load balancing
2. **Dedicated GPU server** (RunPod, Lambda, etc.)
3. **Serverless GPU** (Modal, Replicate)

### Monitoring

**What to Monitor:**
1. **Colab uptime** - Restart if disconnected
2. **ngrok tunnel status** - Update URL if expired
3. **Generation latency** - Should be ~1-2 seconds
4. **Error rate** - Fallback to mock indicates issues

**Logging Example:**
```python
# In homework_service.py
print(f"✅ Generated {len(exercises)} exercises via Colab GPU "
      f"({inference_time:.2f}s, {generated_tokens} tokens)")
```

### Backup Strategy

**Failure Scenarios:**

| Failure | Impact | Automatic Recovery |
|---------|--------|-------------------|
| Colab offline | Falls back to mock | ✅ Yes |
| ngrok expired | Connection timeout | ✅ Falls back to mock |
| Model error | Generation fails | ✅ Falls back to mock |
| Network issue | HTTP timeout | ✅ Falls back to mock |

**No data loss:** All requests fall back gracefully to mock mode.

---

## Alternative Architectures

### Option 1: Local GPU (Not Recommended)

**Pros:**
- No external dependencies
- Lower latency (~0.5s)

**Cons:**
- Requires Mac with GPU (Apple Silicon M1/M2/M3)
- Model won't fit in unified memory (~6GB needed)
- Slower than Colab T4

### Option 2: Cloud GPU (Production Alternative)

**RunPod/Lambda Labs:**
```bash
# Deploy FastAPI service on L4 GPU
# Cost: ~$0.50/hour = ~$360/month (24/7)

# Only worth it for >100 students
```

### Option 3: Serverless (Future)

**Modal, Replicate, etc:**
- Pay per request
- Auto-scaling
- Higher latency (~2-5s)
- More expensive per request

**Recommendation:** Stick with Colab for now. Migrate to cloud GPU when you have >50 active students.

---

## Summary

### Quick Start Checklist

- [ ] Create ngrok account, get auth token
- [ ] Open `demos/colab_inference_api.ipynb` in Colab
- [ ] Enable GPU runtime (T4 free tier)
- [ ] Run all cells, copy ngrok URL
- [ ] `export INFERENCE_API_URL="https://your-url.ngrok.io"`
- [ ] `./run_api.sh`
- [ ] Test with `/api/teacher/assignments`
- [ ] Verify GPU generation in logs

### Key Takeaways

✅ **Free GPU** for development and testing
✅ **4.4x faster** than CPU inference
✅ **Production-ready** with Colab Pro ($10/mo)
✅ **Automatic fallback** to mock mode
✅ **No local GPU** required

### Next Steps

1. **Test integration** - Create a few assignments
2. **Monitor performance** - Check Colab logs
3. **Upgrade to Pro** when needed (longer sessions)
4. **Consider cloud GPU** at >50 students

---

## Support

**Issues?**
- Check [Troubleshooting](#troubleshooting) section
- Review Colab notebook cells for errors
- Check local API logs: `tail -f logs/api.log`
- Verify environment: `echo $INFERENCE_API_URL`

**Questions?**
- Colab docs: https://colab.research.google.com
- ngrok docs: https://ngrok.com/docs
- vLLM docs: https://docs.vllm.ai

---

**Last Updated:** 2025-10-01
**Author:** Ariel Katzir
**Version:** 1.0.0
