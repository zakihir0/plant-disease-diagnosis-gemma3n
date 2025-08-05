#%%
!pip uninstall -y pygobject gradient gradient-utils
!pip install --no-cache-dir --upgrade packaging==24.2
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft==0.16.0 trl triton cut_cross_entropy unsloth_zoo
!pip install --upgrade bitsandbytes
!pip install triton==3.2.0
!pip install pip3-autoremove
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install unsloth==2025.8.1
!pip install faiss-cpu
!pip install sentence-transformers
!pip install wikipedia
# !pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install transformers==4.53.3
!pip install --no-deps --upgrade timm
!pip uninstall -y deepspeed
!pip install deepspeed==0.14.4
!pip install tf-keras
!pip install --upgrade sentence-transformers

#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_COMPILE"] = "0"
os.environ["UNSLOTH_DISABLE_TORCH_COMPILE"] = "1" 

import json
import torch
import torch.nn as nn
import time
from PIL import Image
from tqdm import tqdm
from unsloth import FastModel, FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TextStreamer
from datasets import Dataset, Image as HFImage, load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, AutoProcessor
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import transforms, datasets
import pandas as pd
import random

def get_amp_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

amp_dtype = get_amp_dtype()

# load model and tokenizer (using default precision)
print("🔧 Loading model with default precision (no quantization)...")
llm_model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/gemma-3n-E2B-it",
    dtype=None,  # Default precision (optimized by Unsloth)
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)
print("✅ Model loaded with default precision")

#%%
# Data loading using existing train/valid split
base_dir = '/notebooks/kaggle/input/new_plant_diseases/2/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')

print("📊 Loading dataset...")

def load_dataset_from_dir(root_dir):
    """Load dataset from directory"""
    classes = []
    paths = []
    class_file_counts = {}
    
    for dirname, _, filenames in os.walk(root_dir):
        class_name = dirname.split('/')[-1]
        if class_name == os.path.basename(root_dir):  # Skip root directory
            continue
            
        if class_name not in class_file_counts:
            class_file_counts[class_name] = 0
        
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(dirname, filename)
                if os.path.exists(full_path):
                    classes.append(class_name)
                    paths.append(full_path)
                    class_file_counts[class_name] += 1
    
    return classes, paths, class_file_counts

# Load train and validation datasets
train_classes, train_paths, train_counts = load_dataset_from_dir(train_dir)
val_classes, val_paths, val_counts = load_dataset_from_dir(valid_dir)

# Get unified class names from ImageFolder for consistency
train_dataset_folder = datasets.ImageFolder(root=train_dir)
class_names = train_dataset_folder.classes
print(f"Class names: {class_names}")
print(f"Number of classes: {len(class_names)}")

# Dataset processing for vision classification (gemma3n_plant.py style)
def process_vision_dataset(paths, classes, class_names, dataset_name):
    """Process dataset for vision classification (image path and class ID pairs)"""
    processed_dataset = []
    normal_mapping = {cls: idx for idx, cls in enumerate(class_names)}
    
    for path, class_name in zip(paths, classes):
        class_id = normal_mapping[class_name]
        
        processed_dataset.append({
            "image": path,  # Store image path
            "labels": class_id  # Class ID (int)
        })

    return processed_dataset

# Create dataset (gemma3n_plant.py style)
print("📊 Creating vision datasets...")
train_ds = process_vision_dataset(train_paths, train_classes, class_names, "training")
val_ds = process_vision_dataset(val_paths, val_classes, class_names, "validation")

random.seed(3407)
val_ds = random.sample(val_ds, min(200, len(val_ds)))

print(f"✅ Converted {len(train_ds)} training samples for vision classification")
print(f"✅ Converted {len(val_ds)} validation samples for vision classification")

# Show sample from vision dataset
print("\n📋 Sample vision dataset entry:")
print(f"Image type: {type(train_ds[0]['image'])}")
print(f"Label: {train_ds[0]['labels']} (class: {class_names[train_ds[0]['labels']]})")

#%%
class VisionLinearClassifier(nn.Module):
    def __init__(self, vision_model, num_classes):
        super().__init__()
        self.vision_model = vision_model
        self.pool = nn.AdaptiveAvgPool2d(1)     # (B,C,H,W) → (B,C,1,1)
        hidden = vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, pixel_values, **kwargs):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = self.vision_model(pixel_values.to(torch.bfloat16)).last_hidden_state  # (B,C,H,W)
        feats = self.pool(feats).flatten(1)      # (B,C)
        logits = self.classifier(feats)          # (B,num_classes)
        return logits

class VisionMLPClassifier(nn.Module):
    def __init__(self, vision_model, num_classes, hidden=2048, mlp_dim=512):
        super().__init__()
        self.vision_model = vision_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, pixel_values):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            feats = self.vision_model(pixel_values.to(torch.bfloat16)).last_hidden_state
        feats = self.pool(feats).flatten(1)              # (B,C)
        return self.mlp_head(feats)

class VisionTinyCNN(nn.Module):
    def __init__(self, vision_model, num_classes, hidden=2048):
        super().__init__()
        self.vision_model = vision_model
        self.conv_head = nn.Sequential(
            nn.Conv2d(hidden, 256, 1, bias=False),  # 1x1 dimension reduction
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1, groups=256),  # Depth-wise
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(256, num_classes)

    def forward(self, pixel_values):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x = self.vision_model(pixel_values.to(torch.bfloat16)).last_hidden_state
        x = self.conv_head(x)                # (B,256,H,W)
        x = self.pool(x).flatten(1)          # (B,256)
        return self.fc(x)

# Unified prediction result function
def predict_with_labels(logits, k=3, class_names_list=None):
    """Unified function to get Top-K predictions with class names and probabilities from logits
    
    Args:
        logits (torch.Tensor): Model output logits (tensor or single prediction)
        k (int): Number of top predictions to retrieve (default 3)
        class_names_list (list, optional): List of class names (uses global if None)
    
    Returns:
        list: [{'class_name': str, 'confidence': float}, ...] 
              Direct list for single prediction, list of lists for batch
    """
    
    # Convert 1D logits (single sample) to 2D
    single_prediction = False
    if len(logits.shape) == 1:
        logits = logits.unsqueeze(0)
        single_prediction = True
    
    # Convert to probabilities and get Top-K
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k, dim=-1)
    
    results = []
    for i in range(len(top_k_ids)):
        predictions = []
        for j in range(k):
            class_id = top_k_ids[i][j].item()
            confidence = top_k_probs[i][j].item()
            class_name = class_names_list[class_id]
            predictions.append({
                'class_name': class_name,
                'confidence': confidence
            })
        results.append(predictions)
    
    # Return direct list for single prediction
    return results[0] if single_prediction else results
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

class CNNTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = 1.0  # Gradient clipping norm
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # inputs = {'pixel_values': ..., 'labels': ...}
        # **kwargs handles additional arguments like num_items_in_batch from Unsloth
        logits = model(inputs['pixel_values'])
        loss = F.cross_entropy(logits, inputs['labels'])
        
        return (loss, logits) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        
        pixel_values = inputs.get('pixel_values')
        labels = inputs.get('labels')
        
        with torch.no_grad():
            logits = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
        
        return (loss, logits, labels)

processor = AutoProcessor.from_pretrained(
    "unsloth/gemma-3n-E2B-it",
    trust_remote_code=True
    )
    
# Data collator for vision classification
class VisionDataCollator:
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        # Separate image paths and labels
        image_paths = [item["image"] for item in batch]  # Image paths
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        
        # Process image paths
        pixel_values = []
        for image_path in image_paths:
            # Load PIL Image from path and pass to processor
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            processed = self.processor(
                text="dummy",  # Dummy text
                images=image,  # Pass PIL Image object
                return_tensors="pt"
            )
            pixel_values.append(processed["pixel_values"].squeeze(0))
        
        pixel_values = torch.stack(pixel_values)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

from peft import LoraConfig, get_peft_model, TaskType

vision_tower = llm_model.model.vision_tower
vision_tower.eval(); vision_tower.requires_grad_(False)

lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["qkv", "proj"],   # ViT-based modules
)
lora_tower = get_peft_model(vision_tower, lora_cfg)

vis_model = VisionTinyCNN(vision_tower, num_classes=len(class_names)).to("cuda")

# Training arguments
args = TrainingArguments(
    output_dir="cnn_ckpt",
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,  
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=1e-3,  
    lr_scheduler_type='linear',
    bf16=True, 
    fp16=False,
    remove_unused_columns=False,
    max_steps=100,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True, 
    report_to="none",
    run_name="gemma3n-plant-cnn-finetuning",
    max_grad_norm=1.0, 
)

trainer = CNNTrainer(
    model=vis_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=VisionDataCollator(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

#%%
# ========================================
# CNN Model Inference for Query Generation
# ========================================

def predict_plant_disease(image, vision_model, processor, class_names_list=None):
    """Predict plant disease with CNN model and return class names with probabilities
    
    Args:
        image: PIL Image
        vision_model: Trained Vision model
        processor: Gemma3n processor (AutoProcessor)
        class_names_list: List of class names
    
    Returns:
        list: [{'class_name': str, 'confidence': float}, ...] (Top-3)
    """
    # Convert image to RGB (if needed)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Process image same way as VisionDataCollator
    processed = processor(
        text="dummy",  # Dummy text
        images=image,
        return_tensors="pt"
    )
    pixel_values = processed["pixel_values"].to('cuda')
    
    vision_model.eval()
    
    with torch.no_grad():
        logits = vision_model(pixel_values)
    
    # Get Top-3 predictions
    predictions = predict_with_labels(logits, k=3, class_names_list=class_names_list)
    
    return predictions

print("🔮 CNN inference function ready for Query Generation!")

#%%
# ========================================
# Query Generation Functions
# ========================================

import json
import re
from typing import Any, Dict, List

QUERY_GENERATION_PROMPT = """
# System
You are a plant disease expert and search query generation bot.

# Image Analysis Result (CNN Model Prediction)
Image analysis result: {image_predictions}

# Instructions
1. Based on the above image analysis results (disease diagnosis), convert user's natural language questions or images into suitable database search queries.
2. Utilize CNN model prediction results (disease names and probabilities) to generate more precise and specific queries.
3. Determine whether these queries are plant-related.

# Output Format
## Important: Output only valid JSON

## Exact Format:
{{
  "query": ["<query1>", "<query2>", "<query3>", "<query4>", "<query5>"],
  "is_plant": "<yes or no>"
}}

Examples:
Image analysis result: [{{"class_name": "Tomato___Early_blight", "confidence": 0.85}}, {{"class_name": "Tomato___healthy", "confidence": 0.12}}]
User input: "What is this plant disease?"
Correct output: {{"query":["tomato early blight","tomato disease diagnosis","plant blight symptoms","tomato leaf spots"],"is_plant":"yes"}}

User input: "What's the weather like today?"
Correct output: {{"query":["weather today","current weather"],"is_plant":"no"}}

"""

def generate_response(messages, max_new_tokens=128, temperature=0.7, top_p=0.9, use_no_grad=False):
    """Common function for generating responses from model"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    if use_no_grad:
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, **generation_kwargs)
    else:
        outputs = llm_model.generate(**inputs, **generation_kwargs)
    
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return response

def generate_search_query(text, image_path, vision_model=None, processor=None, class_names_list=None):
    """Generate search query from user natural language input with CNN prediction integration"""

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Analyze image with Vision model (if available)
    image_predictions = []
    if vision_model is not None and processor is not None:
        image_predictions = predict_plant_disease(image, vision_model, processor, class_names_list)
        print(f"🔍 CNN Prediction: {image_predictions}")

    # Embed CNN prediction results into prompt
    prompt_with_predictions = QUERY_GENERATION_PROMPT.format(
        image_predictions=image_predictions if image_predictions else "No analysis results"
    )

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt_with_predictions}
            ]
        },
        {
            "role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text}
            ]
        }
    ]
    
    query = generate_response(
        messages=messages,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9
    ).strip()

    parsed_query = safe_json_loads(query)
    
    return parsed_query

def safe_json_loads(raw: str) -> Dict[str, Any]:
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        # Handle both ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```$", "", cleaned).strip()

    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    data = json.loads(cleaned)

    return data

test1 = "What are the symptoms?"
test2 = "/notebooks/plant_images/apple_apple_scab/image_0.jpg"

# Generate query using Vision model (after training)
parsed_query = generate_search_query(
    text=test1, 
    image_path=test2, 
    vision_model=vis_model,  # Pass trained Vision model
    processor=processor,  # Gemma3n processor
    class_names_list=class_names
)
print(f"🔍 Generated Query with CNN predictions: {parsed_query}")
# %%
# RAG
import os, glob, json, faiss, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DATA_DIR      = "docs"          # Folder to load from
CHUNK_SIZE    = 512              # Approximate character count equivalent to tokens
CHUNK_OVERLAP = 64
INDEX_FILE    = "faiss.index"
META_FILE     = "plant_disease_rag_knowledge_base.json"

def load_and_chunk_doc(path:str):
    """Load file and split by CHUNK_SIZE"""
    chunks = []
    metas = []
    
    # For JSON files, load existing knowledge base
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
        
        for entry in knowledge_base:
            # Split context of each entry
            context = entry['context']
            class_name = entry['class_name']
            
            # Split text by CHUNK_SIZE
            for i in range(0, len(context), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = context[i:i + CHUNK_SIZE]
                if len(chunk_text.strip()) > 50:  # Exclude chunks that are too short
                    chunks.append(chunk_text)
                    metas.append({
                        "source": f"{class_name}",
                        "text": chunk_text,
                        "class_name": class_name,
                        "id": entry['id']
                    })
    
    return chunks, metas

#%%
RAG_DATABASE = [
  {
    "id": "apple_scab_001",
    "class_name": "apple_apple_scab",
    "context": "Apple scab is a common and serious fungal disease affecting apple and crabapple trees. It is caused by the fungus *Venturia inaequalis*. The disease thrives in cool, moist conditions, especially during spring and early summer.\n\n### Symptoms\n\nThe most common signs of apple scab appear on leaves and fruit:\n\n*   **On Leaves:** The first symptoms in spring are often small, velvety, olive-green spots on the undersides of new leaves. These spots eventually darken to a brown or black color and may feel rough or velvety. Infected leaves can become distorted, curled, yellow, and drop prematurely.\n*   **On Fruit:** Infections on young fruit also start as olive-green to black, circular spots. As the fruit develops, these spots become brown, rough, and corky, sometimes causing the fruit to become misshapen and crack. Early infections can cause young fruit to drop. Late-season infections might not be visible at harvest but can develop in storage.\n*   **On Blossoms and Twigs:** The fungus can also infect blossoms, causing them to drop, as well as other parts of the tree like petioles (leaf stems).\n\n### Disease Cycle\n\nThe apple scab fungus overwinters in fallen, infected leaves on the orchard floor. In the spring, during cool, wet weather, fungal spores (ascospores) are released from these dead leaves and are carried by wind and rain to newly emerging leaves and blossoms, starting the primary infection. About 9 to 17 days after infection, these new spots produce a different type of spore (conidia) that can cause secondary infections throughout the summer, especially during rainy periods.\n\n### Management\n\nA combination of cultural practices and chemical control is necessary for effective apple scab management.\n\n#### Cultural Practices:\n\n*   **Sanitation:** Raking and destroying fallen leaves in the autumn or early spring is crucial as it reduces the number of overwintering fungal spores.\n*   **Resistant Cultivars:** Planting scab-resistant apple varieties is an effective long-term management strategy."
  },
  {
    "id": "apple_black_rot_rag_001",
    "class_name": "Apple___Black_rot",
    "context": "Apple Black rot, caused by the fungus *Diplodia seriata* (*Botryosphaeria obtusa*), presents with distinct symptoms on fruit, leaves, and woody tissues, and can be managed through a combination of cultural practices and fungicide applications.\n\nSymptoms:\n*   Fruit: Initial infection on young fruits appears as small, raised purplish lesions. As the fruit ripens, these develop into large, brown, firm lesions that can eventually rot the entire fruit.\n*   Leaves (Frog-eye leaf spot): Symptoms on leaves begin as reddish-brown flecks. These flecks enlarge into circular brown lesions, often surrounded by a purple halo, which can lead to leaf yellowing (chlorosis) and premature leaf drop (abscission).\n*   Cankers: The fungus can also cause cankers on woody tissues. Old cankers, dead shoots, and twig debris serve as sources of inoculum for the disease.\n\nManagement Guide:\nEffective management of Apple Black rot involves several strategies:\n*   Orchard Sanitation: Good orchard sanitation is crucial. This includes removing or thoroughly chopping prunings (e.g., with a flail mower to aid decomposition), removing mummified apples (diseased fruit remaining on the tree or ground), and pruning out dead wood from the trees.\n*   Fire Blight Management: Pruning out current-season shoots infected with fire blight is important because these can be colonized by the black rot pathogen and act as a source of inoculum during the same growing season.\n*   Fungicide Applications: Fungicides used in integrated pest management (IPM) programs can help control *D. seriata*. Specifically, treatments that include strobilurin-containing fungicides (such as Flint, Sovran, Pristine, Luna Sensation, and Merivon) have shown better control of frogeye leaf spot compared to DMI fungicides alone. Reducing primary inoculum on leaf litter on orchard floors, a practice essential for controlling apple scab, can also contribute to black rot management."
  },
  {
    "id": "corn_northern_leaf_blight_rag_001",
    "class_name": "Corn_(maize)___Northern_Leaf_Blight",
    "context": "Northern Corn Leaf Blight (NCLB) is a significant disease in maize production, causing substantial yield reductions, sometimes as high as 30-50% in severe cases.\n\nSymptoms and Identification:\nNCLB primarily affects the corn plant's lamina (leaf blade), though it can also infect the sheath and bract. The disease typically manifests in the middle to later stages of growth, particularly after heading, when temperature and humidity are suitable. Resistant varieties may exhibit brown necrotic stripes along leaf veins with a yellow or hazel chlorotic round around them, producing few or no spores.\n\nManagement Strategies:\n*   Cultural Control:\n    *   Crop rotation with non-corn crops is an effective control method.\n    *   Avoiding planting corn next to small grains or fields with poor grass weed control in the previous year can help prevent issues.\n    *   Early planting can sometimes be beneficial.\n    *   Minimum tillage systems or moist conditions (rain or irrigation) can reduce damage from some pests, which might indirectly impact disease susceptibility.\n    *   Controlling winter annual vegetation (grass and weeds) two weeks before planting is important.\n*   Resistant Varieties: Using resistant varieties has been an effective control measure in the past.\n*   Chemical Control: While the provided information mentions chemical control for other corn issues like insects, it does not specifically detail chemical control for NCLB. However, the general concept of applying pesticides based on infection levels is mentioned in the context of plant disease detection."
  },
  {
    "id": "grape_black_rot_rag_001",
    "class_name": "Grape___Black_rot",
    "context": "Grape Black Rot, caused by the fungus *Guignardia bidwellii*, is a common and damaging disease of grapes, particularly in warm and humid climates. It can lead to significant crop losses if not managed effectively.\n\nSymptoms:\nBlack rot can affect all parts of the grapevine, including leaves, shoots, tendrils, and fruit.\n*   Leaves: Initial symptoms appear as small, yellowish spots that enlarge to form round, tan lesions with dark purple to brown margins. Tiny black dots, which are fungal fruiting bodies (pycnidia), may be visible within these lesions, sometimes arranged in a ring pattern. Severe infection can cause leaf deformity.\n*   Shoots, Petioles, and Tendrils: Lesions on these parts are typically oval, sunken, and purple to black. Black lesions can girdle petioles and shoots, leading to wilting.\n*   Fruit: Fruit symptoms usually appear when grapes are half-grown or pea-sized. Small, light brownish spots form on the berries, which then soften and become sunken. The entire berry quickly rots, shrivels, and turns into hard, black, wrinkled structures called \"mummies,\" which are dotted with pycnidia.\n\nManagement:\nEffective management of grape black rot involves a combination of cultural practices and chemical control.\n\nCultural/Biological Management:\n*   Sanitation: This is the most crucial aspect of black rot management.\n    *   Remove and destroy infected canes, spurs, and fruit clusters (mummies) from the vines.\n    *   Rake up and dispose of fallen leaves and mummies from the vineyard floor.\n    *   Mulch can be used to bury mummies, preventing spores from reaching new grape tissues in the spring.\n*   Pruning: During dormant pruning, remove diseased tendrils and canes to reduce the amount of spores available for infection in the spring.\n*   Site Selection and Air Circulation: Plant grapes in sunny, open areas with good air movement to help dry the plants more quickly, making them less susceptible to black rot. Good weed control is also important."
  },
  {
    "id": "potato_early_blight_rag_001",
    "class_name": "Potato___Early_blight",
    "context": "Potato Early Blight, caused by *Alternaria solani* and *Alternaria alternata*, is a significant disease affecting potato crops worldwide, leading to yield losses of 5-50%. It can damage both the foliage and tubers of potato plants.\n\nSymptoms:\n*   Foliage: Characteristic symptoms include dark brown to black lesions with concentric rings, creating a \"target spot\" effect. These lesions initially appear on older, senescing leaves. They enlarge, coalesce, and can eventually cause leaf death. Spores may be visible on older lesions under a microscope. Lesions can also develop on stems and petioles.\n*   Tubers: Infected tubers develop a dry rot, characterized by isolated, dark, irregular, sunken lesions on the surface. The diseased tissue under these lesions is dark brown, firm, and typically 10-12 mm deep.\n\nManagement:\nEffective management of potato early blight often involves an integrated disease management (IDM) approach, combining cultural practices, resistant cultivars, and fungicide applications.\n\n*   Cultural Practices:\n    *   Crop Rotation: A 3-5 year crop rotation with non-host crops is recommended.\n    *   Sanitation: Field sanitation, including the removal and burning of infected plant debris and eradication of weed hosts, helps reduce inoculum levels.\n    *   Planting Material: Using certified disease-free seed is crucial.\n    *   Nutrition and Water: Providing proper plant nutrition and avoiding water stress can help.\n    *   Volunteer Plants: Destruction of volunteer potato plants in nearby fields throughout the season and destruction of cull piles by freezing or deep burying can prevent disease spread.\n\n*   Resistant Cultivars: Utilizing resistant potato varieties is an important control strategy.\n\n*   Fungicide Application:\n    *   Fungicides are essential for managing early blight, especially under severe conditions.\n    *   Protectant fungicide spray programs are most effective when initiated early in the growing season.\n    *   Proper timing of initial and subsequent applications can reduce the overall frequency of sprays without significant yield loss. Predictive models can be used to time the first application.\n    *   Commonly used fungicides include azoxystrobin, boscalid, difenoconazole, mancozeb, and tebuconazole. Combinations like Azoxystrobin+Flutriafol have shown significant efficacy.\n    *   Ensuring good coverage, especially on lower, senescing leaves, is important during aerial application."
  },
  {
    "id": "apple_cedar_apple_rust_rag_001",
    "class_name": "Apple___Cedar_apple_rust",
    "context": "Cedar-apple rust is a fungal disease caused by *Gymnosporangium juniperi-virginianae*. It has a complex life cycle that requires two host plants: an apple or crabapple tree and a plant from the *Juniperus* genus (e.g., Eastern red cedar).\n\n### Symptoms\n\n*   **On Apple Leaves and Fruit:** The disease first appears in spring as small, pale yellow spots on the upper surfaces of leaves. These spots enlarge, turning a bright orange-yellow color, sometimes with a reddish border. Tiny black fungal bodies (spermogonia) may appear within the spots. Later in the season, distinctive tube-like structures called aecia form on the underside of the leaves, directly beneath the initial spots. Fruit infections are similar, appearing as large orange spots, which can cause the fruit to become deformed and drop prematurely.\n*   **On Cedar/Juniper Host:** On the juniper host, the fungus produces brown, kidney-shaped galls on twigs. In the spring of their second year, during wet weather, these galls swell and produce bright orange, gelatinous 'horns' (telial horns) that release spores to infect nearby apple trees.\n\n### Management\n\nEffective management requires interrupting the disease cycle.\n\n*   **Cultural Control:** The most effective strategy is to remove the alternate host. Eliminating all juniper trees within a one- to two-mile radius of the apple trees can prevent infection, though this is not always practical. Pruning out the cedar galls in late winter or early spring before they produce spores can also reduce the inoculum source.\n*   **Resistant Cultivars:** Planting apple varieties that are resistant to cedar-apple rust is a highly effective and recommended strategy.\n*   **Fungicide Applications:** Protective fungicides can be applied to apple trees to prevent infection. Sprays should begin at the pink bud stage and continue every 7-10 days until about two weeks after petal fall, especially during periods of high rainfall."
  },
  {
    "id": "cherry_powdery_mildew_rag_001",
    "class_name": "Cherry_(including_sour)___Powdery_mildew",
    "context": "Cherry powdery mildew is a fungal disease caused by *Podosphaera clandestina*. It affects leaves, shoots, and fruit, thriving in warm, humid conditions.\n\n### Symptoms\n\n*   **On Leaves:** The disease begins as light-green, circular lesions. These develop into the characteristic white, cotton-like patches of fungal growth. Severely infected leaves can become puckered, twisted, and distorted. Small, black fungal fruiting bodies (chasmothecia) may appear in the white patches later in the season.\n*   **On Fruit:** Fruit infections can be subtle, starting as circular, slightly depressed areas, sometimes with a faint white, powdery growth. The infection often begins near the stem. On ripe fruit, the mildew can appear as a white powdery bloom.\n\n### Management\n\nAn integrated approach is necessary for effective control.\n\n*   **Cultural Control:**\n    *   **Pruning:** Prune trees to improve air circulation and sunlight penetration, which helps reduce humidity.\n    *   **Irrigation:** Use irrigation methods that avoid wetting the foliage.\n    *   **Sucker Removal:** Remove root suckers, as they are highly susceptible to infection.\n    *   **Sanitation:** Rake and remove fallen leaves to reduce the overwintering fungal population.\n*   **Chemical Control:**\n    *   **Fungicides:** Preventative fungicide applications are crucial. A consistent spray program from shuck fall through harvest is often required.\n    *   **Resistance Management:** Rotate fungicides with different modes of action (FRAC groups) to prevent the fungus from developing resistance. Using pre-mixes with multiple modes of action can also be effective.\n    *   **Post-Harvest:** Applying materials like lime sulfur after harvest can help reduce the overwintering inoculum."
  },
  {
    "id": "corn_cercospora_gray_leaf_spot_rag_001",
    "class_name": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "context": "Gray leaf spot (GLS) is a major fungal disease of corn caused by *Cercospora zeae-maydis* and *Cercospora zeina*. It is favored by warm, humid conditions and can cause significant yield loss.\n\n### Symptoms\n\n*   **Initial Lesions:** The first signs are small, tan or olive-green pinpoint spots, sometimes surrounded by a yellow halo. These typically appear on lower leaves first.\n*   **Mature Lesions:** The spots elongate into long, narrow, rectangular lesions that are pale brown to gray. A key characteristic is that the lesions are restricted by the leaf veins, giving them a distinct rectangular or 'matchstick' shape. They can be up to 2.5 inches long.\n*   **Severe Infection:** Under favorable conditions, the lesions can merge, leading to the death of entire leaves. The leaves may take on a grayish appearance due to the production of fungal spores.\n\n### Management\n\nAn integrated management approach is most effective.\n\n*   **Resistant Hybrids:** Planting corn hybrids with genetic resistance to GLS is the most important management strategy.\n*   **Crop Rotation:** Rotating to a non-host crop (e.g., soybeans, wheat) for at least one to two years helps reduce the fungal inoculum in the field, as the fungus overwinters in infected corn residue.\n*   **Residue Management:** Tillage practices that bury or break down corn residue can help reduce the survival of the fungus.\n*   **Fungicides:** Foliar fungicides can be effective, particularly for susceptible hybrids when applied early in the disease development, typically around the tasseling stage. Scouting is important to time applications correctly.\n*   **Cultural Practices:** Avoiding late planting and promoting good air circulation can help reduce disease severity."
  },
  {
    "id": "corn_common_rust_rag_001",
    "class_name": "Corn_(maize)___Common_rust_",
    "context": "Common rust in corn is caused by the fungus *Puccinia sorghi*. The disease is favored by cool, moist weather (61-77°F / 16-25°C). The fungus does not typically overwinter in the main corn-growing regions but its spores are carried north by wind from southern areas each year.\n\n### Symptoms\n\n*   **Pustules:** The most characteristic symptoms are small, oval or elongated, cinnamon-brown to brick-red powdery pustules. These pustules can appear on both the upper and lower surfaces of the leaves, as well as on stalks and husks.\n*   **Development:** The infection starts as small, chlorotic (yellowish) flecks on the leaves which then develop into the pustules. As the season progresses, these pustules may turn black.\n*   **Spores:** The reddish powder in the pustules is composed of fungal spores (urediniospores) that can be easily rubbed off.\n\n### Management\n\nWhile common rust is frequently observed, it rarely causes economic damage in modern hybrid corn.\n*   **Resistant Hybrids:** The primary and most effective management strategy is the use of resistant corn hybrids.\n*   **Fungicides:** Fungicide applications can be effective but are often not necessary for common rust alone. They may be considered if a susceptible hybrid is planted and disease pressure is unusually high early in the season.\n*   **Cultural Practices:** Because the disease is wind-borne and does not rely on local overwintering, practices like crop rotation and tillage are not effective for managing common rust."
  },
  {
    "id": "grape_esca_black_measles_rag_001",
    "class_name": "Grape___Esca_(Black_Measles)",
    "context": "Esca, also known as Black Measles, is a destructive grapevine trunk disease caused by a complex of wood-infecting fungi, primarily *Phaeomoniella chlamydospora* and species of *Phaeoacremonium*. The fungi infect vines through pruning wounds.\n\n### Symptoms\n\nEsca symptoms can be chronic or acute (apoplexy) and affect leaves, fruit, and the woody trunk.\n\n*   **Leaf Symptoms (Chronic):** A characteristic symptom is interveinal 'striping'. In red grape varieties, the stripes are dark red, while in white varieties, they are yellow or chlorotic. These stripes can dry out, and leaves may develop a 'tiger-stripe' pattern. Severely affected leaves may drop prematurely.\n*   **Fruit Symptoms (Chronic):** Berries can develop small, round, dark spots, often bordered by a brown-purple ring, giving them a 'measles' appearance. Affected berries may crack or fail to ripen properly.\n*   **Wood Symptoms:** A cross-section of an infected trunk or cordon reveals dark brown to black streaking or a pattern of dark spots in the vascular tissue (xylem). This is a sign of the internal wood decay.\n*   **Apoplexy (Acute):** This is a sudden collapse of the vine. A part of the vine or the entire plant may rapidly wilt and die, especially during hot summer weather.\n\n### Management\n\nManagement focuses on prevention as there is no cure once a vine is infected.\n\n*   **Pruning Practices:**\n    *   **Wound Protection:** This is the most critical step. Protect large pruning wounds with a wound sealant or paint containing a fungicide (e.g., thiophanate-methyl) to prevent fungal entry.\n    *   **Timing:** Avoid pruning during wet, rainy weather when fungal spores are most active. Delayed pruning (late winter/early spring) can be beneficial.\n*   **Sanitation:**\n    *   **Remove Infected Wood:** Surgically remove and destroy cankered or dead parts of the vine. When removing a diseased trunk, cut well below the visible symptoms of infection.\n*   **Cultural Practices:**\n    *   **Vine Health:** Maintain good vine vigor with proper irrigation and nutrition to help them resist or recover from infection.\n    *   **New Vineyards:** In new plantings, use clean nursery stock and establish a strong vine structure before allowing it to fruit."
  }
]
#%%
# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Load knowledge base and chunk it
docs = []
metas = []

def load_and_chunk_rag_database():
    """Load data from RAG_DATABASE and chunk it"""
    chunks = []
    chunk_metas = []
    
    for entry in RAG_DATABASE:
        context = entry['context']
        class_name = entry['class_name']
        entry_id = entry['id']
        
        # Split context by CHUNK_SIZE
        buf = ""
        for line in context.splitlines():
            if len(buf) + len(line) > CHUNK_SIZE:
                if buf.strip():
                    chunks.append(buf.strip())
                    chunk_metas.append({
                        "source": class_name,
                        "text": buf.strip(),
                        "class_name": class_name,
                        "id": entry_id
                    })
                buf = line[-CHUNK_OVERLAP:] if len(line) > CHUNK_OVERLAP else line
            else:
                buf += " " + line if buf else line
        
        # Add remaining buffer
        if buf.strip():
            chunks.append(buf.strip())
            chunk_metas.append({
                "source": class_name,
                "text": buf.strip(),
                "class_name": class_name,
                "id": entry_id
            })
    
    return chunks, chunk_metas

docs, metas = [], []
print("📄 Loading & chunking from RAG_DATABASE …")
chunks, chunk_metas = load_and_chunk_rag_database()
docs.extend(chunks)
metas.extend(chunk_metas)

#%%
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")
embs  = embedding_model.encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
embs  = np.asarray(embs).astype("float32")   # faiss uses float32

index = faiss.IndexHNSWFlat(embs.shape[1], 32)   # HNSW L=32
index.hnsw.efConstruction = 200
index.add(embs)

print("💾 Index created in memory")

def search(query, k:int=5):
    search_text = " ".join(query["query"])
    
    q_emb = embedding_model.encode([search_text], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, k)    # D: distance, I: index
    hits = []
    
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), 1):
        meta = metas[idx]  # Get directly from metas array in memory
        hits.append({
            "rank": rank,
            "score": float(dist),
            "source": meta["source"],
            "text": meta["text"][:300] + "..." if len(meta["text"]) > 300 else meta["text"],
            "class_name": meta.get("class_name", ""),
            "id": meta.get("id", "")
        })
    return hits

# Execute RAG search
print("🔍 RAG Search Results:")
search_results = search(parsed_query)
for h in search_results:
    print(f"[{h['rank']}] Score: {h['score']:.4f} | {h['source']}")
    print(f"Text: {h['text']}")
    print("-" * 80)

def generate_rag_response(user_question, image_path, vision_model, processor, class_names_list):
    """Comprehensive response generation combining Vision model and RAG search"""
    
    # 1. Image analysis
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    vision_predictions = predict_plant_disease(image, vision_model, processor, class_names_list)
    print(f"🔍 Vision Model Predictions: {vision_predictions}")
    
    # 2. Query generation
    parsed_query = generate_search_query(
        text=user_question,
        image_path=image_path, 
        vision_model=vision_model,
        processor=processor,
        class_names_list=class_names_list
    )
    print(f"📝 Generated Queries: {parsed_query}")
    
    # 3. RAG search
    search_results = search(parsed_query, k=3)
    
    # 4. Context creation
    context_text = "\n".join([f"- {result['text']}" for result in search_results])
    
    # 5. Response generation prompt
    response_prompt = f"""
# Plant Disease Diagnosis System

## Image Analysis Results
{vision_predictions}

## Retrieved Expert Knowledge
{context_text}

## User Question
{user_question}

Based on the above image analysis results and expert knowledge, please provide a detailed and accurate response to the user's question.
Include disease symptoms, management methods, prevention strategies, etc.
"""
    
    # 6. Final response generation
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": response_prompt}
            ]
        }
    ]
    
    final_response = generate_response(
        messages=messages,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.9
    )
    
    return {
        "vision_predictions": vision_predictions,
        "search_results": search_results,
        "final_response": final_response
    }

# Test integrated system
user_question = "What is this plant disease? Please tell me the symptoms and countermeasures."
test_image_path = "/notebooks/plant_images/apple_apple_scab/image_0.jpg"

print("🌱 Plant Disease Diagnosis System - RAG Enhanced")
print("=" * 60)

result = generate_rag_response(
    user_question=user_question,
    image_path=test_image_path,
    vision_model=vis_model,
    processor=processor,
    class_names_list=class_names
)

print("\n📊 Final Results:")
print(f"Vision Predictions: {result['vision_predictions']}")
print(f"\nRetrieved Knowledge: {len(result['search_results'])} relevant documents")
print(f"\nFinal Response:\n{result['final_response']}")

# %%
