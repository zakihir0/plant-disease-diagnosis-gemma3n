# Plant Disease Diagnosis with Gemma3n

A comprehensive plant disease diagnosis system using Gemma3n vision model with CNN classifier and RAG (Retrieval-Augmented Generation) for expert knowledge retrieval.

## 🌟 Features

- **Vision-based Plant Disease Classification**: Uses Gemma3n vision model for accurate plant disease identification
- **Efficient CNN Architecture**: VisionTinyCNN with depth-wise convolutions for optimized processing
- **RAG System**: Retrieval-Augmented Generation with FAISS vector search for comprehensive disease information
- **Intelligent Query Generation**: LLM-powered search query generation from natural language and image inputs
- **Expert Knowledge Base**: Built-in database with detailed information about 9 major plant diseases
- **Interactive Testing**: Comprehensive testing framework with multiple disease scenarios
- **Mixed Precision Training**: Automatic precision selection (BF16/FP16) for optimal performance
- **Comprehensive Jupyter Notebook**: 15-section technical guide for data scientists

## 🏗️ Architecture

### Vision Model
- **VisionTinyCNN**: Lightweight CNN with depth-wise convolutions and frozen vision tower
  - Dimension reduction: 2048 → 256 channels
  - Efficient processing with adaptive pooling
  - Parameter efficiency: Only classification head is trainable

### RAG Pipeline
1. **Image Analysis**: VisionTinyCNN predicts plant disease with Top-3 confidence scores
2. **Query Generation**: LLM converts user questions + predictions into structured search queries
3. **Knowledge Retrieval**: FAISS HNSW semantic search through expert disease database
4. **Response Generation**: Comprehensive answers combining vision predictions and retrieved knowledge

### Testing Features
- **Multi-Disease Testing**: Automated testing with Apple scab, Potato early blight, and Corn northern leaf blight
- **Complete Response Display**: Full expert recommendations with symptoms and management strategies
- **Accuracy Validation**: Comparison between predicted and expected disease classifications

## 📋 Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for training)

### Key Libraries
- `unsloth==2025.8.1` - Efficient model training and inference
- `transformers==4.53.3` - Hugging Face transformers
- `faiss-cpu` - FAISS vector similarity search
- `sentence-transformers` - Text embeddings for RAG
- `Pillow` - Image processing
- `matplotlib` - Visualization and image display
- `scikit-learn` - Metrics and evaluation

### Complete Dependencies
See `requirements.txt` for the full list of dependencies with specific versions.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/zakihir0/plant-disease-diagnosis-gemma3n.git
cd plant-disease-diagnosis-gemma3n
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For CUDA support, install PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 🚀 Usage

### Quick Start

1. **Run the complete system**:
```python
# Execute workflow_2.py for full training and demo
python workflow_2.py
```

2. **Or use Jupyter Notebook**:
```bash
jupyter notebook plant_disease_diagnosis_notebook.ipynb
```

### API Usage

1. **Load trained model**:
```python
from unsloth import FastVisionModel
from transformers import AutoProcessor

# Load Gemma3n vision model with mixed precision
def get_amp_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/gemma-3n-E2B-it",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)
```

2. **Train VisionTinyCNN classifier**:
```python
# Create efficient CNN classifier
vision_tower = model.model.vision_tower
vis_model = VisionTinyCNN(vision_tower, num_classes=len(class_names))

# Training with FP16 precision
args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,
    fp16=True,  # Mixed precision training
    max_steps=100,
    learning_rate=1e-3,
)
```

3. **Generate comprehensive diagnosis**:
```python
# Full RAG pipeline
result = generate_rag_response(
    user_question="What is this plant disease?",
    image_path="path/to/plant_image.jpg",
    vision_model=vis_model,
    processor=processor,
    class_names_list=class_names
)

# Get comprehensive expert response
print(f"Diagnosis: {result['final_response']}")
print(f"Vision Predictions: {result['vision_predictions']}")
```

### Supported Plant Diseases

The system includes knowledge about various plant diseases including:
- Apple diseases (scab, black rot, cedar apple rust)
- Corn diseases (northern leaf blight, gray leaf spot, common rust)
- Grape diseases (black rot, esca)
- Potato early blight
- Cherry powdery mildew

## Model Architecture

- **Base Model**: Gemma3n-E2B-it with vision capabilities
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient training
- **Vision Processing**: Frozen vision tower with trainable classification head
- **Knowledge Base**: FAISS index with sentence transformers embeddings

## Performance

- **Memory Efficient**: Uses 4-bit quantization and gradient checkpointing
- **Fast Inference**: Optimized with Unsloth acceleration
- **Accurate**: Combines vision predictions with expert knowledge
- **Scalable**: FAISS-based search for large knowledge bases

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient model training
- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Gemma](https://ai.google.dev/gemma) for the base vision model

## Citation

If you use this work in your research, please cite:

```bibtex
@software{plant_disease_diagnosis_gemma3n,
  title={Plant Disease Diagnosis with Gemma3n},
  author={Your Name},
  year={2025},
  url={https://github.com/zakihir0/plant-disease-diagnosis-gemma3n}
}
```