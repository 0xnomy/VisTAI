"""
Prompt templates for medical image chat interactions.
PRODUCTION VERSION - NO MOCKING, NO PLACEHOLDERS

These templates define the EXACT text that the LLM receives.
NO IMAGES. NO EMBEDDINGS. ONLY STRUCTURED TEXT FACTS.
"""

from typing import Dict, List, Optional


class SystemPrompt:
    """System-level instructions for the LLM - STRICT MEDICAL AI CONSTRAINTS."""
    
    @staticmethod
    def get_medical_assistant_prompt() -> str:
        """
        System prompt defining the assistant as a medical image analysis explainer.
        
        This prompt FORCES the LLM to:
        1. Explain ONLY provided CNN predictions (no speculation)
        2. Refuse diagnosis, treatment, or clinical advice
        3. Cite facts explicitly from the analysis
        
        Returns:
            System prompt string (immutable)
        """
        return """You explain AI bone tumor predictions. You are NOT a doctor.

RULES:
- Answer concisely in 1-3 sentences
- Do NOT repeat context or instructions
- Do NOT echo analysis summaries
- Do NOT interpret severity, prognosis, or normality unless explicitly stated
- Only restate facts relevant to the question
- NEVER give medical advice or treatment recommendations
- End with: "AI prediction requires medical validation."

Answer only what was asked."""


class ChatPromptTemplate:
    """
    Template for building chat prompts with structured facts.
    ALL LLM INPUTS MUST GO THROUGH THESE TEMPLATES.
    """
    
    @staticmethod
    def build_initial_analysis(facts: Dict) -> str:
        """
        Build prompt for initial image analysis summary.
        
        This converts structured CNN outputs into a natural language prompt
        that the LLM can use to generate a summary. The LLM receives ONLY
        these text facts - no images, no embeddings.
        
        Args:
            facts: Structured facts from CNN inference containing:
                   - tumor_class (str): Primary prediction
                   - confidence (float): 0-1 confidence score
                   - malignancy (str): 'malignant' or 'benign'
                   - tumor_coverage (float): Percentage from segmentation mask
                   - tumor_location (str): Spatial description
                   - alternative_diagnoses (list): Top-3 alternatives
            
        Returns:
            Formatted prompt string (deterministic, no variability)
        """
        # Build concise fact summary
        parts = []
        parts.append(f"Detected {facts.get('tumor_class', 'unknown')} with {facts.get('confidence', 0) * 100:.0f}% confidence")
        
        if 'malignancy' in facts:
            parts.append(f"classified as {facts['malignancy']}")
        
        if 'tumor_location' in facts:
            parts.append(f"located in {facts['tumor_location']}")
        
        if 'tumor_coverage' in facts:
            parts.append(f"covering {facts['tumor_coverage']:.1f}% of the image")
        
        prompt = "Summarize these AI findings in 2-3 sentences:\n\n"
        prompt += ". ".join(parts) + ".\n\n"
        
        if 'alternative_diagnoses' in facts and len(facts['alternative_diagnoses']) > 0:
            alts = ", ".join([f"{cls} ({conf*100:.0f}%)" for cls, conf in facts['alternative_diagnoses'][:2]])
            prompt += f"Alternative possibilities: {alts}.\n\n"
        
        prompt += "State facts concisely. End with disclaimer."
        
        return prompt
    
    @staticmethod
    def build_qa_prompt(facts: Dict, question: str, context: Optional[str] = None) -> str:
        """
        Build prompt for question-answering interactions.
        
        This is the CORE grounding mechanism. The LLM receives:
        1. Structured facts (text) from CNN predictions
        2. User's question
        3. Optional retrieved context (text summaries of similar cases)
        
        The LLM does NOT receive:
        - Raw images
        - CLIP embeddings
        - Segmentation masks
        - Class logits
        
        This structure prevents hallucination by limiting the LLM's knowledge
        to explicitly provided, verifiable facts.
        
        Args:
            facts: Structured facts from CNN inference (REQUIRED)
            question: User's natural language question (REQUIRED)
            context: Optional retrieved case summaries (text only)
            
        Returns:
            Formatted prompt string (deterministic structure)
        """
        # Build minimal context (facts only, no formatting that gets echoed)
        context_lines = []
        context_lines.append(f"Tumor: {facts.get('tumor_class', 'unknown')}")
        context_lines.append(f"Confidence: {facts.get('confidence', 0) * 100:.0f}%")
        
        if 'malignancy' in facts:
            context_lines.append(f"Type: {facts['malignancy']}")
        
        if 'tumor_location' in facts:
            context_lines.append(f"Location: {facts['tumor_location']}")
        
        if 'tumor_coverage' in facts:
            context_lines.append(f"Coverage: {facts['tumor_coverage']:.1f}%")
        
        if 'alternative_diagnoses' in facts and len(facts['alternative_diagnoses']) > 0:
            alts = ', '.join([f"{cls} ({conf*100:.0f}%)" for cls, conf in facts['alternative_diagnoses'][:2]])
            context_lines.append(f"Alternatives: {alts}")
        
        prompt = "CONTEXT: " + " | ".join(context_lines) + "\n\n"
        prompt += f"QUESTION: {question}\n\n"
        prompt += "Answer in 1-3 sentences using only the context above."
        
        return prompt


# Module exports
__all__ = ['SystemPrompt', 'ChatPromptTemplate']
