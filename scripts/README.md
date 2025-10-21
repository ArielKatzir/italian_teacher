# Scripts Directory

Utility scripts for Italian Teacher project setup and management.

## 🔽 Model Download Script

Downloads and caches Llama models for local use.

### Quick Start

```bash
# With your HF token
python scripts/download_models.py --token hf_your_token_here

# Using environment variable
export HF_TOKEN=hf_your_token_here
python scripts/download_models.py

# Interactive mode (choose models)
python scripts/download_models.py
```

### Available Models

- **llama3.1-3b** - Fast, good for development (~6GB, ~2GB RAM with 4-bit)
- **llama3.1-8b** - Best quality/speed balance (~16GB, ~5GB RAM with 4-bit)
- **mistral-7b** - Efficient multilingual (~14GB, ~4GB RAM with 4-bit)

### Options

```bash
# List available models
python scripts/download_models.py --list

# Download specific model
python scripts/download_models.py --model llama3.1-3b

# Download all models
python scripts/download_models.py --all

# Force re-download
python scripts/download_models.py --force --model llama3.1-3b
```

### Requirements

```bash
pip install torch transformers huggingface_hub bitsandbytes accelerate
```

### GPU Memory Requirements

- **Tesla T4 (16GB)**: Can run llama3.1-8b with 4-bit quantization
- **Lower memory GPUs**: Use llama3.1-3b or mistral-7b
- **CPU only**: All models work (slower)

### Troubleshooting

**"No valid model identifier"**:
- Make sure you have a valid HF token
- Accept the model license at https://huggingface.co/meta-llama/Llama-3.1-3B-Instruct

**"Permission denied"**:
- Get a new token with "Read" permissions
- Make sure token is not expired

**"Out of memory"**:
- Use smaller model (3B instead of 8B)
- Ensure 4-bit quantization is enabled
- Close other GPU applications