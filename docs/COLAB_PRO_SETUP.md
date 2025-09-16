# Google Colab Pro Setup Guide for Italian Teacher

## 🎯 **Why Colab Pro is Perfect for Italian Teacher**

**Colab Pro gives you everything you need:**
- 🔥 **Free GPU access** (T4, V100, A100) - perfect for 7B-8B language models
- 💾 **High RAM** (25-51GB) - enough for quantized large models
- 🖥️ **Terminal access** - run CLI applications directly
- 💽 **Google Drive integration** - seamless project access
- 🚀 **No local setup** - everything runs in the cloud
- 💰 **Cost effective** - $10/month vs $600+ for API usage

**Perfect for MVP development and testing!**

## 🚀 Quick Start (5 Minutes)

### 1. Open New Colab Notebook
- Go to [Google Colab](https://colab.research.google.com/)
- Create new notebook or open existing one
- **Change to GPU runtime**: Runtime → Change runtime type → GPU

### 2. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Navigate to Project
```bash
%%bash
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
ls -la  # Verify project files are there
```

### 4. Install Dependencies
```bash
%%bash
pip install transformers torch accelerate bitsandbytes pydantic
```

### 5. Run Italian Teacher
```python
import subprocess
import os

# Change to project directory
os.chdir('/content/drive/MyDrive/Colab Notebooks/italian_teacher')

# Run the CLI (note: interactive mode works in terminal)
print("🇮🇹 Starting Italian Teacher CLI...")
print("📱 Use the terminal for interactive chat or run in background")
```

### 6. Interactive Terminal Method
```bash
# Open terminal in Colab (click terminal icon or use shortcut)
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
python cli/simple_chat.py
```

## 🤖 Optimal Model Selection for Colab Pro

### Automatic Selection (Recommended)
The CLI automatically detects Colab and selects the best model:

**Colab Pro T4 GPU (15GB):**
```python
# Automatically selected: Mistral 7B with 4-bit quantization
# Perfect balance of quality and memory usage
```

**Colab Pro A100 GPU (40GB):**
```python
# Automatically selected: Llama 3.1 8B with 4-bit quantization
# Maximum quality for Italian conversations
```

### Manual Override (Advanced)
```python
# Force specific model for testing
from models import create_model

# Lightweight for testing
model = create_model("llama3.1-3b", device="cuda", quantization="4bit")

# High quality (needs 16GB+ GPU memory)
model = create_model("llama3.1-8b", device="cuda", quantization="4bit")

# Balanced option
model = create_model("mistral-7b", device="cuda", quantization="4bit")
```

## 📊 Performance Monitoring

### Check GPU Status
```python
# Check GPU availability and memory
!nvidia-smi

# Check PyTorch GPU access
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
```

### Monitor Memory Usage
```python
# During model loading/usage
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

print_gpu_memory()
```

### Clear Memory When Needed
```python
# If you get CUDA out of memory errors
import torch
torch.cuda.empty_cache()
print("🧹 GPU memory cleared")
```

## 🛠️ Development Workflow

### 1. Code Development
```python
# Edit files directly in Colab or use your local editor
# Files sync automatically with Google Drive

# Test changes
%cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
!python -m pytest tests/unit/test_marco_agent.py -v
```

### 2. Model Testing
```python
# Quick model test
%cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
!python -c "
from models import create_model
import asyncio

async def test():
    model = create_model('mistral-7b', device='cuda', quantization='4bit')
    await model.load_model()
    response = await model.generate_response('Ciao! Come stai?')
    print(f'Response: {response.text}')

asyncio.run(test())
"
```

### 3. Interactive Chat Testing
```bash
# In Colab terminal
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
python cli/simple_chat.py
```

## 🚨 Common Issues & Solutions

### "CUDA out of memory"
```python
# Solution 1: Clear memory
import torch
torch.cuda.empty_cache()

# Solution 2: Restart runtime
# Runtime → Restart runtime

# Solution 3: Use smaller model
# The CLI will automatically fallback to smaller models
```

### "Module not found"
```bash
# Reinstall dependencies
pip install --upgrade transformers torch accelerate bitsandbytes pydantic
```

### "Permission denied" or file access issues
```python
# Remount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Check file permissions
import os
os.chmod('/content/drive/MyDrive/Colab Notebooks/italian_teacher/cli/simple_chat.py', 0o755)
```

### "Model download too slow"
```python
# Use environment variable to speed up downloads
import os
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'  # Show progress
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'     # Faster downloads

# Or use mock model for testing
model = create_model("llama3.1-8b", mock=True)  # Instant loading
```

## 🔄 Colab Pro Limitations & Workarounds

### Session Timeouts
**Issue**: Colab sessions timeout after ~12 hours of inactivity
**Solution**:
```python
# Keep session alive (run in background)
import time
import random

def keep_alive():
    while True:
        time.sleep(random.randint(300, 600))  # 5-10 minutes
        print("🔄 Keeping session alive...")

# Run in background (optional)
import threading
threading.Thread(target=keep_alive, daemon=True).start()
```

### GPU Availability
**Issue**: GPU might not always be available
**Solution**:
```python
# Automatic fallback to CPU
if not torch.cuda.is_available():
    print("⚠️ GPU not available, using CPU model")
    model = create_model("llama3.1-3b", device="cpu")
```

### File Persistence
**Issue**: Files in `/content/` are lost when session ends
**Solution**:
```python
# Always work in Google Drive
os.chdir('/content/drive/MyDrive/Colab Notebooks/italian_teacher')
# Files are automatically saved to Google Drive
```

## 📈 Performance Optimization Tips

### Model Loading Optimization
```python
# Cache models to avoid repeated downloads
import os
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/.cache/transformers'
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'
```

### Memory Efficient Training
```python
# Use gradient checkpointing for future fine-tuning
model_config = {
    "gradient_checkpointing": True,
    "use_cache": False,  # Saves memory during training
    "quantization": "4bit"
}
```

### Faster Inference
```python
# Optimize for inference speed
torch.backends.cudnn.benchmark = True  # Faster for fixed input sizes
torch.set_num_threads(4)  # Optimize CPU usage
```

## 💰 Cost Optimization

### Maximize Free Usage
- Use GPU when you need it, CPU for development
- Close unused notebooks to free up resources
- Use mock models for testing, real models for validation

### Colab Pro Tips
```python
# Check current usage
!nvidia-smi  # Monitor GPU usage
!df -h       # Check disk usage
!free -h     # Check RAM usage

# Optimize resource usage
# Use quantization to reduce memory
# Close unnecessary browser tabs
# Use terminal for long-running tasks
```

## 🎯 Recommended Workflow

### Daily Development
1. **Morning**: Start Colab, mount drive, install dependencies
2. **Development**: Edit code, run tests with mock models
3. **Testing**: Load real model, test conversations
4. **Evening**: Save progress, clear GPU memory

### User Testing
1. **Load real model** (Mistral 7B or Llama 8B)
2. **Run CLI** for interactive testing
3. **Collect feedback** from 5-10 users
4. **Iterate quickly** with immediate model access

### Production Preparation
1. **Test multiple models** to find optimal quality/speed
2. **Measure performance** (response time, memory usage)
3. **Document setup** for future deployment
4. **Export successful configurations**

**Bottom Line**: Colab Pro gives you a powerful, cost-effective development environment that's perfect for testing and validating your Italian Teacher MVP!