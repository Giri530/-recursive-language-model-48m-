# 🔄 Recursive Language Model (48M)

A novel transformer-based language model featuring **adaptive recursive processing** for intelligent text generation. Unlike traditional transformers, this model learns to dynamically allocate computational resources based on input complexity.

## 🌟 Key Features

- **Adaptive Computation**: Router network decides optimal recursion depth per input
- **Efficient Architecture**: 48M parameters with competitive performance
- **Fast Inference**: Optimized for resource-constrained environments
- **Easy Integration**: Compatible with Hugging Face Transformers
- **Short Training Time**: Trained in just 2.24 hours on a single T4 GPU

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Girinath11/recursive-language-model-48m.git
cd recursive-language-model-48m

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, GPT2Tokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Girinath11/recursive-language-model-48m",
    trust_remote_code=True
)
tokenizer = GPT2Tokenizer.from_pretrained("Girinath11/recursive-language-model-48m")

# Generate text
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

outputs = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.8,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🏗️ Architecture

### Model Specifications

| Component | Details |
|-----------|---------|
| **Parameters** | 47,931,907 (~48M) |
| **Vocabulary Size** | 50,257 tokens (GPT-2) |
| **Embedding Dimension** | 512 |
| **Transformer Layers** | 6 base layers |
| **Attention Heads** | 8 heads |
| **Max Recursion Steps** | 2 |
| **Context Length** | 256 tokens |

### Architecture Overview

```
Input Text
    ↓
Token & Position Embeddings (512d)
    ↓
Transformer Stack (6 layers)
    ↓
    ├─→ Router Network → Recursion Depth Decision
    ↓
Recursive Processing Layer (adaptive passes)
    ↓
Language Model Head (50,257 vocab)
    ↓
Output Text
```

**Key Innovation**: The router network learns to predict optimal recursion depth, allowing the model to spend more computation on complex inputs while being efficient on simpler ones.

## 📊 Performance

### Training Results

| Metric | Value |
|--------|-------|
| **Final Validation Loss** | 3.84 |
| **Final Perplexity** | 46.75 |
| **Training Time** | 2.24 hours |
| **Training Hardware** | NVIDIA T4 (16GB) |
| **Training Samples** | 100,000 documents |

### Training Progression

| Epoch | Train Loss | Val Loss | Perplexity |
|-------|-----------|----------|------------|
| 1 | 7.38 | 6.01 | 406.28 |
| 2 | 5.50 | 4.97 | 143.59 |
| 4 | 4.28 | 4.15 | 63.62 |
| 6 | 3.81 | 3.90 | 49.27 |
| 8 | 3.59 | 3.84 | **46.75** |

### Inference Speed

| Hardware | Tokens/Second |
|----------|--------------|
| CPU (Intel i7) | ~80-120 |
| GPU (T4) | ~400-600 |
| GPU (V100) | ~700-1000 |

## 💻 Advanced Usage

### Temperature Control

```python
# More creative generation
creative_output = model.generate(
    input_ids, 
    temperature=1.0, 
    max_new_tokens=100
)

# More focused generation
focused_output = model.generate(
    input_ids, 
    temperature=0.5, 
    max_new_tokens=100
)
```

### Batch Generation

```python
prompts = [
    "The history of computers",
    "Climate change impacts",
    "Recent AI breakthroughs"
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True)
outputs = model.generate(**inputs, max_new_tokens=50)

for i, output in enumerate(outputs):
    print(f"Prompt: {prompts[i]}")
    print(f"Output: {tokenizer.decode(output, skip_special_tokens=True)}\n")
```

## 🎯 Use Cases

### ✅ Recommended Applications

- Educational tools and learning systems
- Research on adaptive computation
- Prototyping language model applications
- Resource-constrained deployment
- Text completion experiments

### ❌ Not Recommended For

- Production chatbots without human oversight
- Applications requiring high factual accuracy
- Professional writing without verification
- Medical, legal, or financial advice

## ⚠️ Limitations

- **Context Window**: 256 tokens (relatively short)
- **Model Size**: 48M parameters (smaller than production models)
- **Repetition**: May repeat phrases in long generations
- **Factuality**: Can generate plausible but incorrect information
- **Domain Knowledge**: Limited specialized/technical knowledge

## 📝 Citation

If you use this model in your research, please cite:

```bibtex
@misc{girinath2025recursive_language_model,
  author = {Girinath V},
  title = {Recursive Language Model with Adaptive Depth Processing},
  year = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Girinath11/recursive-language-model-48m}}
}
```

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Framework**: PyTorch and Hugging Face Transformers
- **Inspiration**: Adaptive Computation Time and Mixture of Experts research
- **Training Infrastructure**: Kaggle/Colab GPU resources
- **Community**: Hugging Face community for feedback and support

## 📧 Contact

- **Author**: Girinath V
- **Hugging Face**: [@Girinath11](https://huggingface.co/Girinath11)
- **Issues**: [GitHub Issues](https://github.com/Girinath11/recursive-language-model-48m/issues)
- **Discussions**: [Model Discussion Board](https://huggingface.co/Girinath11/recursive-language-model-48m/discussions)

---

**⭐ Star this repository if you find it helpful!**

**📢 Share with others who might benefit from adaptive language models!**
