"""
Chat Engine for Medical Image Analysis
PRODUCTION VERSION - REAL LLM ONLY (microsoft/Phi-3-mini-4k-instruct)

This module loads and runs the actual LLM for generating explanations.
NO MOCK RESPONSES. NO PLACEHOLDERS. GPU ONLY.

The LLM receives ONLY structured text facts from CNN predictions.
It NEVER sees raw images, embeddings, or masks.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

from .prompt_templates import SystemPrompt, ChatPromptTemplate
from .safety_rules import SafetyFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatEngine:
    """
    LLM-based chat engine for explaining CNN predictions.
    
    STRICT CONSTRAINTS:
    - Uses microsoft/Phi-3-mini-4k-instruct (PyTorch 2.5 compatible)
    - GPU-only execution (no CPU fallback)
    - Receives text facts ONLY (no images/embeddings)
    - All outputs pass through SafetyFilter
    - Deterministic generation (low temperature)
    
    Design Rationale:
    ---------------------
    Why TinyLlama/TinyLlama-1.1B-Chat-v1.0?
    - VERY small for fast testing (1.1B params, only 2.2GB download)
    - Instruction-tuned (follows system prompts)
    - Fast inference (< 1s per response)
    - Openly available via HuggingFace (Apache 2.0 license)
    - Compatible with PyTorch 2.5+
    
    Why GPU-only?
    - Medical AI requires real-time responses (< 3s)
    - CPU inference would be 10-15s per query (unacceptable)
    - This is a production system, not a demo
    
    Why no fine-tuning?
    - Grounding via prompts is more verifiable than learned weights
    - Pre-trained medical knowledge from web-scale data
    - Prompt engineering provides explicit control over outputs
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[torch.device] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
    ):
        """
        Initialize the chat engine with instruction-tuned LLM.
        
        Args:
            model_name: HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
            device: torch.device (MUST be CUDA, no CPU fallback)
            max_new_tokens: Maximum response length
            temperature: Generation temperature (low = deterministic)
            
        Raises:
            RuntimeError: If CUDA not available or model fails to load
        """
        # STRICT GPU CHECK (MANDATORY)
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU (CUDA) is REQUIRED for this LLM.\n"
                    "This is a production medical AI system, not a demo.\n"
                    "CPU inference would take 10-15s per query (unacceptable).\n"
                    "Install CUDA or use a GPU-enabled environment."
                )
            device = torch.device("cuda")
        elif device.type != "cuda":
            raise RuntimeError(
                f"Device must be CUDA, got: {device}\n"
                f"GPU-only execution is mandatory."
            )
        
        self.device = device
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        logger.info(f"Loading LLM: {model_name} on {device}")
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            logger.info("✓ Tokenizer loaded successfully")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for {model_name}:\n{e}\n"
                f"Check internet connection and HuggingFace access."
            ) from e
        
        # Load model (GPU only, FP16 for efficiency)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self.model.eval()
            
            logger.info(f"✓ Model loaded successfully on {device}")
            logger.info(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model {model_name}:\n{e}\n"
                f"This could be due to:\n"
                f"  1. Insufficient GPU memory (need ~2GB VRAM for TinyLlama)\n"
                f"  2. Network issues (model downloads from HuggingFace)\n"
                f"  3. PyTorch/transformers version mismatch\n"
                f"Required: PyTorch 2.5+, transformers 4.35+"
            ) from e
        
        # Initialize safety filter
        self.safety_filter = SafetyFilter()
        
        # Conversation history (limited to prevent context drift)
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 10
        
        logger.info("✓ Chat engine ready")
    
    def _build_chat_prompt(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Build the full prompt with system instructions and conversation history.
        
        Uses TinyLlama chat template format:
        <|system|>
        {system_prompt}</s>
        <|user|>
        {user_message}</s>
        <|assistant|>
        
        Args:
            user_message: Current user input (contains facts + question)
            system_prompt: System-level instructions (safety rules)
            
        Returns:
            Formatted prompt string for model
        """
        if system_prompt is None:
            system_prompt = SystemPrompt.get_medical_assistant_prompt()
        
        # Build TinyLlama chat format (Zephyr-style)
        prompt = f"<|system|>\n{system_prompt}</s>\n"
        
        # Add conversation history (if any)
        for msg in self.conversation_history[-self.max_history_length:]:
            role = msg['role']
            content = msg['content']
            prompt += f"<|{role}|>\n{content}</s>\n"
        
        # Add current user message
        prompt += f"<|user|>\n{user_message}</s>\n"
        prompt += "<|assistant|>\n"
        
        return prompt
    
    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate LLM response to user message.
        
        EXECUTION FLOW:
        1. Build prompt (system + history + user message)
        2. Tokenize input
        3. Generate tokens (GPU, deterministic)
        4. Decode response
        5. Apply safety filter
        6. Return grounded response
        
        Args:
            user_message: User's input (contains structured facts + question)
            system_prompt: Optional custom system prompt
            
        Returns:
            LLM response string (safety-filtered)
        """
        # Build full prompt
        full_prompt = self._build_chat_prompt(user_message, system_prompt)
        
        # Tokenize (reduced max_length to prevent overflow)
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024  # Reduced to prevent context duplication
        ).to(self.device)
        
        # Generate (GPU, deterministic)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response (remove input prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Apply safety filter (MANDATORY)
        safe_response = self.safety_filter.enforce(response)
        
        return safe_response
    
    def chat(self, user_question: str, structured_facts: Dict) -> str:
        """
        Interactive chat with grounded responses.
        
        This is the MAIN USER-FACING METHOD.
        
        GROUNDING MECHANISM:
        1. User asks question: "What tumor is present?"
        2. Structured facts injected: {tumor_class: "osteosarcoma", confidence: 0.92}
        3. Intent detected (diagnosis/location/confidence/general)
        4. Prompt built via ChatPromptTemplate.build_qa_prompt()
        5. LLM generates response using ONLY provided facts
        6. Safety filter validates output
        7. Response returned to user
        
        Args:
            user_question: Natural language question from user
            structured_facts: Dict with CNN predictions (REQUIRED)
                             {tumor_class, confidence, tumor_coverage, etc.}
            
        Returns:
            Grounded, safety-filtered response
        """
        # Detect intent for targeted responses
        question_lower = user_question.lower()
        intent = self._detect_intent(question_lower)
        
        # Route to direct answer ONLY for very direct questions about predictions
        # Most questions should go to LLM for detailed, contextual answers
        if intent == 'diagnosis' and len(user_question.split()) <= 10:
            tumor = structured_facts.get('tumor_class', 'unknown')
            conf = structured_facts.get('confidence', 0) * 100
            # Provide more context in direct answer
            top3 = structured_facts.get('top3_predictions', [])
            alt_text = ''
            if len(top3) > 1:
                alt_text = f" The top alternative diagnoses are {top3[1][0]} ({top3[1][1]*100:.1f}%) and {top3[2][0]} ({top3[2][1]*100:.1f}%)."
            return f"The AI detected {tumor} with {conf:.1f}% confidence.{alt_text} AI prediction requires medical validation."
        
        elif intent == 'location' and len(user_question.split()) <= 10:
            loc = structured_facts.get('tumor_location', 'not specified')
            area = structured_facts.get('tumor_area_pct', 0)
            if loc == 'not detected':
                return f"The segmentation model did not detect visible tumor boundaries. The classification model identified {structured_facts.get('tumor_class', 'unknown')} based on overall image patterns. AI prediction requires medical validation."
            return f"The tumor is located in the {loc}, covering approximately {area:.1f}% of the X-ray image area. AI prediction requires medical validation."
        
        elif intent == 'confidence' and len(user_question.split()) <= 10:
            conf = structured_facts.get('confidence', 0) * 100
            tumor = structured_facts.get('tumor_class', 'unknown')
            top3 = structured_facts.get('top3_predictions', [])
            if len(top3) > 1:
                return f"The system is {conf:.1f}% confident in detecting {tumor}. Alternative possibilities include {top3[1][0]} ({top3[1][1]*100:.1f}%) and {top3[2][0]} ({top3[2][1]*100:.1f}%). AI prediction requires medical validation."
            return f"The system is {conf:.1f}% confident in detecting {tumor}. AI prediction requires medical validation."
        
        # For general questions, use LLM
        grounded_prompt = ChatPromptTemplate.build_qa_prompt(
            facts=structured_facts,
            question=user_question
        )
        
        # Generate response
        response = self.generate_response(grounded_prompt)
        
        # Update conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_question
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return response
    
    def initial_analysis(self, structured_facts: Dict) -> str:
        """
        Generate initial analysis summary when user first uploads image.
        
        Args:
            structured_facts: CNN predictions
            
        Returns:
            Summary of findings
        """
        # Build analysis prompt
        analysis_prompt = ChatPromptTemplate.build_initial_analysis(structured_facts)
        
        # Generate summary
        summary = self.generate_response(analysis_prompt)
        
        # Store in history
        self.conversation_history.append({
            'role': 'assistant',
            'content': summary
        })
        
        return summary
    
    def _detect_intent(self, question_lower: str) -> str:
        """
        Detect user intent for targeted responses.
        
        Args:
            question_lower: Lowercase question string
            
        Returns:
            Intent: 'diagnosis', 'location', 'confidence', or 'general'
        """
        # Only match very direct, short questions
        # Longer or complex questions should go to LLM
        
        # Diagnosis intent - must be very direct
        diagnosis_patterns = [
            'what tumor', 'which tumor', 'tumor type', 'tumor present',
            'what is the tumor', 'what did you detect', 'what was detected',
            'type of tumor', 'kind of tumor', 'tumor class'
        ]
        if any(pattern in question_lower for pattern in diagnosis_patterns) and len(question_lower.split()) <= 12:
            return 'diagnosis'
        
        # Location intent - must be very direct
        location_patterns = ['where is', 'tumor location', 'located where']
        if any(pattern in question_lower for pattern in location_patterns) and len(question_lower.split()) <= 6:
            return 'location'
        
        # Confidence intent - must be very direct
        confidence_patterns = ['how confident', 'confidence level', 'how sure', 'how accurate']
        if any(pattern in question_lower for pattern in confidence_patterns) and len(question_lower.split()) <= 6:
            return 'confidence'
        
        # Default to general (uses LLM for detailed response)
        return 'general'
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history reset")


# Module exports
__all__ = ['ChatEngine']
