# Plant Disease Diagnosis with Gemma3n

A comprehensive plant disease diagnosis system using Gemma3n vision model with CNN classifiers and RAG (Retrieval-Augmented Generation) for knowledge retrieval.

## Features

- **Vision-based Plant Disease Classification**: Uses Gemma3n vision model for accurate plant disease identification
- **CNN Model Training**: Multiple classifier architectures including Linear, MLP, and TinyCNN with LoRA fine-tuning
- **RAG System**: Retrieval-Augmented Generation for comprehensive disease information and management recommendations
- **Query Generation**: Intelligent search query generation from natural language and image inputs
- **Expert Knowledge Base**: Built-in database with detailed information about various plant diseases

## Architecture

### Vision Models
- **VisionLinearClassifier**: Simple linear classifier on top of frozen vision features
- **VisionMLPClassifier**: Multi-layer perceptron with dropout for better generalization
- **VisionTinyCNN**: Lightweight CNN with depth-wise convolutions for efficient processing

### RAG Pipeline
1. **Image Analysis**: CNN model predicts plant disease with confidence scores
2. **Query Generation**: Converts user questions and predictions into search queries
3. **Knowledge Retrieval**: FAISS-based semantic search through disease database
4. **Response Generation**: Comprehensive answers using retrieved knowledge and vision predictions

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### ML/AI Libraries
- `unsloth>=2025.8.1` - Efficient model training and inference
- `transformers>=4.53.3` - Hugging Face transformers
- `torch` - PyTorch framework
- `torchvision` - Computer vision utilities
- `peft` - Parameter-efficient fine-tuning
- `trl` - Transformer reinforcement learning

### Vision & Processing
- `Pillow` - Image processing
- `timm` - Image models
- `datasets` - Dataset utilities

### RAG & Search
- `faiss-cpu` - Similarity search
- `sentence-transformers` - Text embeddings
- `numpy` - Numerical computations

### Training & Evaluation
- `accelerate` - Distributed training
- `scikit-learn` - Metrics and evaluation
- `pandas` - Data manipulation
- `tqdm` - Progress bars

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

## Usage

### Basic Usage

1. **Load the model and setup**:
```python
from unsloth import FastVisionModel
from transformers import AutoProcessor

# Load Gemma3n vision model
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/gemma-3n-E2B-it",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)

processor = AutoProcessor.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    trust_remote_code=True
)
```

2. **Train a classifier**:
```python
# Create and train vision classifier
vision_tower = model.model.vision_tower
vis_model = VisionTinyCNN(vision_tower, num_classes=len(class_names))

trainer = CNNTrainer(
    model=vis_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=VisionDataCollator(processor),
)

trainer.train()
```

3. **Use RAG for diagnosis**:
```python
# Generate comprehensive diagnosis
result = generate_rag_response(
    user_question="What is this plant disease?",
    image_path="path/to/plant_image.jpg",
    vision_model=vis_model,
    processor=processor,
    class_names_list=class_names
)

print(f"Diagnosis: {result['final_response']}")
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