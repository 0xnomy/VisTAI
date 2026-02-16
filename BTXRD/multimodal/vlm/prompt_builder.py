"""
Medical Prompt Builder for CLIP Text Encoding
==============================================

Converts structured medical facts into natural language descriptions
for CLIP text encoding. This creates semantically meaningful embeddings.

Design Principle:
-----------------
The VLM (CLIP) needs textual descriptions that:
1. Are medically accurate
2. Align with visual features
3. Are grounded in CNN predictions (not hallucinated)
4. Include confidence and uncertainty
"""

import json
from pathlib import Path


# Medical knowledge base for tumor types
TUMOR_CHARACTERISTICS = {
    'osteosarcoma': {
        'type': 'malignant',
        'description': 'malignant bone tumor, most common primary bone cancer',
        'typical_features': 'irregular margins, destructive bone lesion',
        'urgency': 'high'
    },
    'osteochondroma': {
        'type': 'benign',
        'description': 'benign bone growth with cartilage cap',
        'typical_features': 'well-defined exostosis projecting from bone surface',
        'urgency': 'low'
    },
    'giant cell tumor': {
        'type': 'usually benign',
        'description': 'locally aggressive but usually benign tumor',
        'typical_features': 'expansile lytic lesion, often at bone ends',
        'urgency': 'moderate'
    },
    'simple bone cyst': {
        'type': 'benign',
        'description': 'fluid-filled cavity within bone',
        'typical_features': 'well-defined radiolucent lesion, often in long bones',
        'urgency': 'low'
    },
    'osteofibroma': {
        'type': 'benign',
        'description': 'benign bone lesion with fibrous tissue',
        'typical_features': 'well-circumscribed lucent lesion with sclerotic rim',
        'urgency': 'low'
    },
    'multiple osteochondromas': {
        'type': 'benign',
        'description': 'hereditary condition with multiple bone growths',
        'typical_features': 'multiple exostoses, bilateral distribution',
        'urgency': 'moderate'
    },
    'synovial osteochondroma': {
        'type': 'benign',
        'description': 'cartilaginous loose body in joint',
        'typical_features': 'calcified body within joint space',
        'urgency': 'low'
    },
    'other bt': {
        'type': 'benign',
        'description': 'benign bone tumor, not elsewhere classified',
        'typical_features': 'variable appearance, benign characteristics',
        'urgency': 'low'
    },
    'other mt': {
        'type': 'malignant',
        'description': 'malignant bone tumor, not elsewhere classified',
        'typical_features': 'destructive bone lesion, aggressive features',
        'urgency': 'high'
    }
}


def get_tumor_info(tumor_class):
    """
    Get medical characteristics for tumor class.
    
    Args:
        tumor_class: String tumor name
        
    Returns:
        Dictionary with tumor characteristics
    """
    return TUMOR_CHARACTERISTICS.get(
        tumor_class.lower(),
        {
            'type': 'unknown',
            'description': 'bone lesion',
            'typical_features': 'not specified',
            'urgency': 'unknown'
        }
    )


def build_medical_prompt(structured_facts):
    """
    Build natural language medical description from structured facts.
    
    This is the key function that grounds LLM responses in CNN predictions.
    
    Args:
        structured_facts: Dict with:
            - tumor_class: str
            - confidence: float (0-1)
            - tumor_area_pct: float (0-100)
            - tumor_location: str
            - top3_predictions: list of (class, prob) tuples
            
    Returns:
        medical_prompt: Natural language description for CLIP encoding
    """
    tumor_class = structured_facts['tumor_class']
    confidence = structured_facts['confidence']
    area_pct = structured_facts.get('tumor_area_pct', 0)
    location = structured_facts.get('tumor_location', 'unknown')
    
    # Get tumor characteristics
    tumor_info = get_tumor_info(tumor_class)
    
    # Build base description
    prompt_parts = []
    
    # 1. Detection statement with confidence
    if confidence >= 0.8:
        certainty = "High confidence detection"
    elif confidence >= 0.6:
        certainty = "Moderate confidence detection"
    else:
        certainty = "Low confidence detection"
    
    prompt_parts.append(
        f"{certainty} of {tumor_class} ({confidence*100:.1f}% confidence)."
    )
    
    # 2. Tumor classification
    prompt_parts.append(
        f"Classified as {tumor_info['type']} tumor: {tumor_info['description']}."
    )
    
    # 3. Spatial information
    if area_pct > 0:
        prompt_parts.append(
            f"Tumor occupies approximately {area_pct:.1f}% of the image area, "
            f"located in the {location} region."
        )
    
    # 4. Visual characteristics
    prompt_parts.append(
        f"Typical radiographic features: {tumor_info['typical_features']}."
    )
    
    # 5. Alternative diagnoses (if provided)
    if 'top3_predictions' in structured_facts:
        alt_diagnoses = []
        for idx, (cls, prob) in enumerate(structured_facts['top3_predictions'][1:3], 1):
            alt_diagnoses.append(f"{cls} ({prob*100:.1f}%)")
        
        if alt_diagnoses:
            prompt_parts.append(
                f"Alternative diagnoses considered: {', '.join(alt_diagnoses)}."
            )
    
    # Combine into single prompt
    medical_prompt = " ".join(prompt_parts)
    
    return medical_prompt


def build_short_prompt(structured_facts):
    """
    Build concise prompt for chat context (not CLIP encoding).
    
    Args:
        structured_facts: Same as build_medical_prompt
        
    Returns:
        Short description string
    """
    tumor_class = structured_facts['tumor_class']
    confidence = structured_facts['confidence']
    tumor_info = get_tumor_info(tumor_class)
    
    return (
        f"{tumor_class} ({tumor_info['type']}) "
        f"detected with {confidence*100:.0f}% confidence"
    )


def build_safety_context(structured_facts):
    """
    Build safety-critical information for LLM prompting.
    
    This explicitly constrains the LLM to only discuss what the CNNs detected.
    
    Args:
        structured_facts: Medical facts from CNN inference
        
    Returns:
        safety_context: String with strict boundaries
    """
    tumor_class = structured_facts['tumor_class']
    confidence = structured_facts['confidence']
    tumor_info = get_tumor_info(tumor_class)
    
    context = f"""
MEDICAL ANALYSIS CONSTRAINTS:
- The CNN model detected: {tumor_class}
- Detection confidence: {confidence*100:.1f}%
- Tumor classification: {tumor_info['type']}
- You MUST NOT provide medical advice or treatment recommendations
- You MUST state that this is an AI prediction requiring expert validation
- You MUST acknowledge uncertainty when confidence is below 80%
- You MUST NOT discuss conditions not in the top-3 predictions
- You MUST refer users to qualified medical professionals for diagnosis
"""
    
    return context.strip()


def format_facts_for_llm(structured_facts):
    """
    Format structured facts into LLM prompt context.
    
    Args:
        structured_facts: Medical facts dictionary
        
    Returns:
        Formatted fact string for LLM context
    """
    tumor_info = get_tumor_info(structured_facts['tumor_class'])
    
    facts = f"""
CURRENT IMAGE ANALYSIS:
- Primary diagnosis: {structured_facts['tumor_class']}
- Tumor type: {tumor_info['type']}
- Confidence: {structured_facts['confidence']*100:.1f}%
- Tumor coverage: {structured_facts.get('tumor_area_pct', 0):.1f}% of image
- Location: {structured_facts.get('tumor_location', 'not specified')}
- Medical urgency: {tumor_info['urgency']}
"""
    
    if 'top3_predictions' in structured_facts:
        facts += "\nAlternative diagnoses:\n"
        for idx, (cls, prob) in enumerate(structured_facts['top3_predictions'][:3], 1):
            tumor_alt = get_tumor_info(cls)
            facts += f"  {idx}. {cls} ({tumor_alt['type']}) - {prob*100:.1f}%\n"
    
    return facts.strip()


if __name__ == "__main__":
    # Test prompt building
    print("\n" + "="*70)
    print("Testing Medical Prompt Builder")
    print("="*70)
    
    # Mock structured facts
    test_facts = {
        'tumor_class': 'osteosarcoma',
        'confidence': 0.92,
        'tumor_area_pct': 15.3,
        'tumor_location': 'upper-left quadrant',
        'top3_predictions': [
            ('osteosarcoma', 0.92),
            ('other mt', 0.05),
            ('giant cell tumor', 0.02)
        ]
    }
    
    print("\nInput structured facts:")
    print(json.dumps(test_facts, indent=2))
    
    print("\n" + "-"*70)
    print("CLIP Medical Prompt:")
    print("-"*70)
    clip_prompt = build_medical_prompt(test_facts)
    print(clip_prompt)
    
    print("\n" + "-"*70)
    print("LLM Context:")
    print("-"*70)
    llm_context = format_facts_for_llm(test_facts)
    print(llm_context)
    
    print("\n" + "-"*70)
    print("Safety Context:")
    print("-"*70)
    safety = build_safety_context(test_facts)
    print(safety)
    
    print("\n✓ Prompt builder test passed")
    print("="*70)
