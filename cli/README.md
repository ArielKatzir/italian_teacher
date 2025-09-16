# CLI Interface - Complete Italian Teacher Chat

This directory contains the complete CLI interface for chatting with Marco using the full Italian Teacher agent system with language model integration.

## 🚀 Quick Start

### 1. Local Development
```bash
# From the project root directory
cd italian_teacher
source ~/.venvs/py312/bin/activate
python cli/simple_chat.py
```

### 2. Google Colab Pro (Recommended)
```bash
# In Colab terminal
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
python cli/simple_chat.py
```

## 🤖 Automatic Model Selection

The CLI automatically detects your environment and selects the best model:

- **🔥 Colab Pro + GPU**: Llama 3.1 8B or Mistral 7B with 4-bit quantization
- **💻 Local + GPU**: Optimized model based on VRAM
- **🏠 Local CPU**: Smaller Llama 3.1 3B model
- **🧪 No PyTorch**: Mock model for testing

## 📊 Colab Pro Advantages

**Perfect for Italian Teacher development:**
- ✅ **Free GPU access** (T4/A100) - run 8B models easily
- ✅ **High RAM** (25-51GB) - load large models with quantization
- ✅ **Terminal access** - run CLI directly
- ✅ **Google Drive integration** - seamless file access
- ✅ **No local setup** - everything runs in the cloud

## 🎭 Complete Features

This CLI uses the **full Marco agent** with:
- ✅ **Language Model Integration**: Real LLM responses (Llama/Mistral)
- ✅ **Personality System**: Marco's friendly, encouraging nature
- ✅ **Error Tolerance**: Grammar correction with patience settings
- ✅ **Motivation Tracking**: Progress monitoring and encouragement
- ✅ **Cultural Context**: Italian cultural knowledge integration
- ✅ **Educational Features**: Homework help and learning guidance

## 📱 Example Session

```
🇮🇹 Welcome to Italian Teacher - Complete Marco Chat!
Type 'quit', 'exit', or 'ciao' to end the conversation.
============================================================
🤖 Selecting optimal model for your environment...
📊 Detected Google Colab environment
🚀 GPU detected: Tesla T4 (15.8GB)
⚡ Using Mistral 7B - optimized for your GPU memory
🔄 Initializing language model...
✅ Language model loaded successfully!

✅ Marco agent initialized with full personality system!
🎭 Features active: Error correction, motivation tracking, cultural context
📱 Starting conversation...

🇮🇹 Marco: Ciao! Sono Marco, il tuo amico italiano! I'm excited to help you practice Italian today. What would you like to talk about?

You: Ciao Marco! I need help with my Italian homework about ordering food at a restaurant
🇮🇹 Marco: Perfetto! Restaurant ordering is such a practical skill. Let's practice together! First, when you enter a restaurant, you might hear "Buonasera, quanti siete?" - that means "Good evening, how many are you?" How would you respond if you're dining alone?

You: Sono solo?
🇮🇹 Marco: Molto bene! Almost perfect! You'd say "Sono solo" (I'm alone) or more commonly "Un tavolo per uno, per favore" (A table for one, please). Your Italian is great! What would you like to order? Should we practice with appetizers, main courses, or drinks?
```

## 🛠️ Setup for Colab Pro

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to Project
```bash
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
```

### 3. Install Dependencies (if needed)
```bash
pip install transformers torch accelerate bitsandbytes
```

### 4. Run Italian Teacher
```bash
python cli/simple_chat.py
```

## 🔧 Model Configuration

### Automatic Selection (Recommended)
The CLI automatically chooses the best model for your environment. No configuration needed!

### Manual Model Selection
```python
# For testing without GPU
model = create_model("llama3.1-8b", mock=True)

# For Colab Pro with T4 GPU
model = create_model("mistral-7b", device="cuda", quantization="4bit")

# For Colab Pro with A100 GPU
model = create_model("llama3.1-8b", device="cuda", quantization="4bit")
```

## 📈 Performance Tips

### Colab Pro Optimization
- **Use GPU runtime**: Runtime → Change runtime type → GPU
- **Enable 4-bit quantization**: Reduces memory usage by ~75%
- **Monitor GPU memory**: `!nvidia-smi` to check usage
- **Restart if needed**: Runtime → Restart runtime to clear memory

### Memory Management
```python
# Clear GPU memory if needed
import torch
torch.cuda.empty_cache()
```

## 🚨 Troubleshooting

### "CUDA out of memory"
- Try smaller model: The CLI will automatically fallback
- Restart runtime: Runtime → Restart runtime
- Check GPU usage: `!nvidia-smi`

### "Module not found"
```bash
# Install missing dependencies
pip install transformers torch accelerate bitsandbytes pydantic
```

### "Model loading failed"
- Check internet connection (models download from HuggingFace)
- Fallback to mock model will be used automatically
- Try restarting runtime and running again

## 🎯 What This Tests

**Real Italian Learning Experience:**
- 🗣️ **Natural conversation** with Marco's personality
- 📚 **Homework assistance** with explanations
- ✏️ **Grammar correction** with encouragement
- 🇮🇹 **Cultural context** in responses
- 📊 **Learning progress** tracking
- 🤖 **AI model performance** on Italian language tasks

## 🔮 Next Steps

1. **Test with real users**: Get 5-10 people to try conversations
2. **Collect feedback**: What works well? What needs improvement?
3. **Measure engagement**: How long do people chat? Do they come back?
4. **Iterate on personality**: Adjust Marco's responses based on feedback

This CLI is your **complete MVP** - ready for real user testing!