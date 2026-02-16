# VLM-Powered Medical Image Chat System

**A multimodal reasoning layer for bone tumor X-ray analysis**

---

## 🎯 Project Overview

This system extends the existing bone tumor classification/segmentation pipeline with:

1. **Vision-Language Model (VLM)** encoding for semantic alignment
2. **Language Model (LLM)** chat interface for interactive Q&A
3. **Strict safety constraints** to prevent medical misinformation
4. **Retrieval-augmented generation** for contextual responses

**CRITICAL**: This system does NOT replace or retrain the existing CNN models. It provides an explanatory layer on top of frozen predictions.

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────┐
│         USER UPLOADS X-RAY IMAGE                │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────▼───────────┐
       │  EXISTING CNN LAYER   │
       │  (Frozen, Untouched)  │
       └─────┬─────────┬───────┘
             │         │
    ┌────────▼──┐  ┌──▼────────┐
    │ EfficientNet│ │MobileNetV2│
    │ Classifier │  │  UNet     │
    └────────┬───┘  └──┬────────┘
             │         │
             └─────┬───┘
                   │
         ┌─────────▼──────────┐
         │ STRUCTURED FACTS   │
         │ • tumor_class      │
         │ • confidence       │
         │ • tumor_location   │
         │ • tumor_area       │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   VLM ENCODER      │
         │   (CLIP, Frozen)   │
         │ • Image embedding  │
         │ • Text embedding   │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │   LLM CHAT ENGINE  │
         │ (Phi-2/TinyLlama)  │
         │ • NO raw images    │
         │ • Only facts       │
         │ • Safety enforced  │
         └─────────┬──────────┘
                   │
                   ▼
         ┌─────────────────┐
         │ GROUNDED ANSWER │
         └─────────────────┘
```

---

## 🔑 Key Design Principles

### 1. **CNNs Are Frozen**
- EfficientNet-B0 classification model: **NEVER retrained**
- MobileNetV2-UNet segmentation model: **NEVER retrained**
- All parameters frozen (`requires_grad = False`)
- Models used only for inference

### 2. **VLM Is a Frozen Encoder**
- Uses OpenCLIP (RN50 or ViT variants)
- **NOT used for classification or segmentation**
- Creates aligned image-text embeddings
- Enables semantic similarity search
- All parameters frozen (no fine-tuning)

### 3. **LLM Never Sees Raw Images**
- Receives ONLY structured facts from CNNs
- No visual features or pixels
- Prevents hallucination beyond CNN outputs
- Grounded in measurable predictions

### 4. **Strict Safety Enforcement**
- Cannot provide medical advice
- Cannot make definitive diagnoses
- Must acknowledge uncertainty
- Must refer to medical professionals
- Filters prohibited language patterns

---

## 📦 Module Structure

```
BTXRD/
├── multimodal/                    # NEW MODULES (do not modify existing)
│   ├── __init__.py
│   ├── pipeline.py               # Main orchestrator
│   │
│   ├── vlm/                      # Vision-Language Model
│   │   ├── __init__.py
│   │   ├── clip_encoder.py      # Frozen CLIP encoder
│   │   ├── prompt_builder.py    # Medical prompt generation
│   │   └── embedding_utils.py   # Embedding operations
│   │
│   ├── llm/                      # Language Model
│   │   ├── __init__.py
│   │   ├── chat_engine.py       # LLM inference
│   │   ├── prompt_templates.py  # Prompt templates
│   │   └── safety_rules.py      # Safety constraints
│   │
│   └── retrieval/                # Similarity Search
│       ├── __init__.py
│       ├── csv_index.py         # CSV metadata indexing
│       └── similarity_search.py # Embedding-based retrieval
│
├── multimodal_chat_demo.py       # Demo script
│
├── classification/                # EXISTING (unchanged)
├── segmentation/                  # EXISTING (unchanged)
├── common/                        # EXISTING (unchanged)
└── inference.py                   # EXISTING (unchanged)
```

---

## 🚀 Usage

### Basic Usage

```python
from multimodal.pipeline import create_pipeline

# Initialize pipeline
pipeline = create_pipeline(
    project_root='/path/to/BTXRD',
    vlm_model='RN50',
    llm_model='microsoft/phi-2'  # or None for mock LLM
)

# Process X-ray image
result = pipeline.process_image('path/to/xray.jpg')

# Get initial summary
print(pipeline.get_initial_summary())

# Interactive chat
response = pipeline.chat("What tumor is present?")
print(response)

response = pipeline.chat("How confident is the model?")
print(response)
```

### Command-Line Demo

```bash
# With mock LLM (no heavy model loading)
python multimodal_chat_demo.py --image path/to/xray.jpg

# With real LLM (requires GPU + transformers)
python multimodal_chat_demo.py --image path/to/xray.jpg --llm microsoft/phi-2

# Different VLM architectures
python multimodal_chat_demo.py --image path/to/xray.jpg --vlm ViT-B-32

# Analysis only (no chat)
python multimodal_chat_demo.py --image path/to/xray.jpg --no-chat
```

---

## 🔬 How Each Component Works

### 1. **CNN Inference** (`pipeline.run_cnn_inference`)

```python
# Input: X-ray image path
# Output: Structured facts dictionary

structured_facts = {
    'tumor_class': 'osteosarcoma',
    'confidence': 0.92,
    'tumor_area_pct': 15.3,
    'tumor_location': 'upper-left quadrant',
    'top3_predictions': [
        ('osteosarcoma', 0.92),
        ('other mt', 0.05),
        ('giant cell tumor', 0.02)
    ],
    'segmentation_mask': numpy.ndarray
}
```

### 2. **VLM Encoding** (`pipeline.encode_with_vlm`)

```python
# Input: Image + structured facts
# Output: Aligned embeddings

# Step 1: Encode image
image_emb = vlm_encoder.encode_image(image_path)  # [512-dim]

# Step 2: Build medical prompt from facts
medical_prompt = build_medical_prompt(structured_facts)
# → "High confidence detection of osteosarcoma (92.0% confidence). 
#    Classified as malignant tumor: most common primary bone cancer..."

# Step 3: Encode medical prompt
text_emb = vlm_encoder.encode_text(medical_prompt)  # [512-dim]

# Step 4: Combine embeddings
combined_emb = combine_embeddings(image_emb, text_emb)  # [512-dim]
```

**Why this is VLM-powered:**
- Image and medical facts share the same embedding space
- Enables semantic similarity: "osteosarcoma X-ray" ≈ "malignant bone tumor"
- Retrieval: Find similar historical cases
- Grounding: Visual features aligned with textual descriptions

### 3. **LLM Chat** (`pipeline.chat`)

```python
# Input: User question + structured facts
# Output: Grounded, safe response

# User asks: "What tumor is present?"

# LLM receives ONLY:
context = """
CURRENT IMAGE ANALYSIS:
- Detected: osteosarcoma
- Confidence: 92.0%
- Alternative diagnoses: other mt (5%), giant cell tumor (2%)

USER QUESTION: What tumor is present?

Answer based ONLY on the CNN analysis above.
"""

# LLM generates response
response = llm.generate(context)

# Safety filter enforces constraints
safe_response = safety_filter.enforce(response, confidence=0.92)

# Final output:
# "Based on the CNN analysis, the model detected osteosarcoma with 92%
#  confidence. This is classified as a malignant bone tumor. This 
#  prediction requires validation by a qualified medical professional..."
```

---

## 🛡️ Safety Mechanisms

### Safety Rule Enforcement

1. **Prohibited Patterns**
   - ✗ "You should get treatment..."
   - ✗ "I recommend surgery..."
   - ✗ "Don't worry, it's benign..."
   - ✗ "The diagnosis is confirmed..."

2. **Required Elements**
   - ✓ "This is an AI prediction"
   - ✓ "Requires validation by medical professional"
   - ✓ Acknowledge uncertainty if confidence < 80%

3. **Question Filtering**
   - Block: "Should I get surgery?"
   - Block: "What treatment do I need?"
   - Allow: "What tumor is present?"
   - Allow: "How confident is the model?"

### Implementation

```python
from multimodal.llm.safety_rules import SafetyFilter, filter_user_question

# Check user question
is_allowed, reason = filter_user_question("Should I get treatment?")
# → (False, "Cannot provide medical recommendations")

# Enforce response safety
filter = SafetyFilter(strict_mode=True)
safe_response = filter.enforce(response, confidence=0.65)
# → Adds disclaimers and uncertainty language
```

---

## 🎓 Why This Is "VLM-Powered"

### What the VLM DOES:

1. **Semantic Alignment**: Maps images and text to shared embedding space
2. **Retrieval Enhancement**: Enables similarity search over historical cases
3. **Grounding**: Visual features correlated with medical descriptions
4. **Explainability**: VLM embeddings show what textual concepts align with image regions

### What the VLM Does NOT Do:

1. ✗ Replace CNN classification
2. ✗ Replace CNN segmentation
3. ✗ Make diagnostic decisions
4. ✗ Generate medical labels
5. ✗ Fine-tune on medical data

### Technical Justification:

**VLM Role**: Create a **multimodal representation space** where:
- Medical images have semantic meaning (not just pixel grids)
- Textual facts have visual grounding (not just tokens)
- LLM responses can reference both modalities coherently

**Example**:
```python
# Image embedding
img_emb = vlm.encode_image(xray)  # [512-dim vector]

# Text embedding
text_emb = vlm.encode_text("malignant bone tumor with irregular margins")

# Similarity
sim = compute_similarity(img_emb, text_emb)  # 0.78 (high alignment)

# This similarity indicates the image visually exhibits characteristics
# described by "malignant bone tumor" → provides evidence for LLM response
```

---

## 🖥️ Edge Deployment

### Recommended Configuration

**For Edge Devices (e.g., hospital workstations):**

```python
pipeline = create_pipeline(
    project_root='/path/to/BTXRD',
    vlm_model='RN50',           # 38M params, 512-dim embeddings
    llm_model='microsoft/phi-2', # 2.7B params with 4-bit quantization
    device=torch.device('cuda')  # or 'cpu' for CPU-only
)
```

**Memory Footprint:**
- CNNs (frozen): ~7MB + ~5MB = 12MB
- CLIP RN50: ~150MB
- Phi-2 (4-bit): ~1.5GB
- **Total**: ~1.7GB VRAM

**Latency:**
- CNN inference: ~50ms (GPU)
- CLIP encoding: ~20ms
- LLM generation: ~1-2s (GPU with quantization)
- **Total**: ~2-3s per query

---

## 📊 Example Chat Session

```
📊 IMAGE ANALYSIS COMPLETE

PRIMARY FINDING:
• Tumor Type: osteosarcoma
• Confidence: 87.0%
• Tumor Coverage: 16.2% of image
• Location: distal femur region

ALTERNATIVE DIAGNOSES:
  2. other mt (8.0%)
  3. giant cell tumor (3.0%)

⚠️ IMPORTANT:
This is an AI-generated analysis. All findings must be validated
by a qualified medical professional before any clinical decisions.

💬 You can now ask questions about this analysis.

---

💬 YOU: What tumor is present in this image?

🤖 ASSISTANT:
Based on the CNN analysis, the model detected osteosarcoma with 87% confidence.
Osteosarcoma is classified as a malignant bone tumor and is the most common
primary bone cancer. The model identified typical radiographic features 
including a destructive bone lesion with irregular margins.

This is an AI prediction requiring expert validation by a qualified radiologist
or orthopedic specialist for clinical decision-making.

---

💬 YOU: How confident is the model?

🤖 ASSISTANT:
The classification model reports 87% confidence, which indicates high certainty.
This confidence level suggests the model has strong evidence for the osteosarcoma
classification based on visual features in the X-ray.

However, the model also considered alternative diagnoses including other 
malignant tumors (8%) and giant cell tumor (3%), indicating some diagnostic
uncertainty. Additional imaging studies and expert review are recommended 
to confirm the diagnosis.

This is an AI prediction that requires validation by a medical professional.

---

💬 YOU: Where is the tumor located?

🤖 ASSISTANT:
The segmentation model identified the tumor in the distal femur region,
occupying approximately 16.2% of the image area. The distal femur (lower
thigh bone near the knee) is a common location for osteosarcoma, particularly
in adolescents and young adults.

Accurate tumor localization requires expert radiographic interpretation and
correlation with clinical findings. This automated analysis should be confirmed
by a qualified medical professional.
```

---

## 🔧 Dependencies

### Core Requirements
```bash
pip install torch torchvision
pip install open-clip-torch
pip install transformers accelerate bitsandbytes
pip install pandas numpy pillow
```

### Optional (for testing)
```bash
pip install pytest
pip install matplotlib  # for visualizations
```

---

## 🧪 Testing

### Test Individual Modules

```bash
# Test CLIP encoder
python multimodal/vlm/clip_encoder.py

# Test prompt builder
python multimodal/vlm/prompt_builder.py

# Test safety rules
python multimodal/llm/safety_rules.py

# Test pipeline
python multimodal/pipeline.py
```

### Integration Test

```bash
# Run on test image
python multimodal_chat_demo.py --image segmentation_test.csv[0]
```

---

## 📖 Citation

If you use this multimodal extension, please cite both the original BTXRD project and the VLM/LLM components:

```bibtex
@software{btxrd_multimodal_2024,
  title={VLM-Powered Medical Image Chat for Bone Tumor Analysis},
  author={BTXRD Team},
  year={2024},
  note={Multimodal extension of BTXRD classification/segmentation pipeline}
}
```

---

## ⚠️ Medical Disclaimer

**THIS SOFTWARE IS FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- NOT approved for clinical use
- NOT a substitute for professional medical advice
- NOT a diagnostic tool
- Requires validation by qualified medical professionals
- Users assume all risks and responsibilities

---

## 🤝 Contributing

To extend this system:

1. **Do NOT** modify existing CNN models
2. **Do NOT** retrain VLM or change its architecture
3. **Do** enhance prompt templates for better responses
4. **Do** improve safety filters
5. **Do** add retrieval capabilities
6. **Do** optimize for edge deployment

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Contact: btxrd-team@example.com

---

**Built with ❤️ for safe, explainable medical AI**
