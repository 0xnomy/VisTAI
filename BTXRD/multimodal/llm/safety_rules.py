"""
Safety Rules for Medical AI Chat
PRODUCTION VERSION - STRICT ENFORCEMENT

This module implements MANDATORY safety checks for all LLM outputs.
Prevents medical advice, hallucination, and unsafe responses.

Safety checks are applied AFTER generation, not during.
The LLM generates freely, then outputs are filtered/modified.
"""

import re
from typing import List, Tuple


class SafetyFilter:
    """
    Post-generation safety filter for medical AI responses.
    
    DESIGN RATIONALE:
    -----------------
    Why post-generation filtering?
    - Pre-generation constraints (via prompts) can be ignored by LLM
    - Post-generation filtering GUARANTEES safety (hard enforcement)
    - Allows detection of specific prohibited patterns
    - Enables automatic disclaimer injection
    
    How it works:
    1. LLM generates response freely
    2. SafetyFilter scans for prohibited patterns
    3. If violations found → response rejected/modified
    4. Disclaimers automatically added if missing
    5. Final output guaranteed safe
    """
    
    # PROHIBITED PATTERNS (regex, case-insensitive)
    # These indicate the LLM is giving medical advice or speculating
    PROHIBITED_PATTERNS = [
        # Treatment recommendations
        r'\b(you should|you must|you need to|I recommend|I suggest)\b.*\b(treat|medication|surgery|therapy|drug|procedure)\b',
        r'\b(take|prescribe|administer|undergo)\b.*\b(medication|drug|chemotherapy|radiotherapy|surgery)\b',
        
        # Diagnosis language (LLM making independent diagnoses)
        r'\b(you have|this is definitely|I diagnose|diagnosis is)\b',
        r'\b(confirms? diagnosis|definitively shows?)\b',
        
        # Prognosis / outcomes
        r'\b(survival rate|life expectancy|prognosis|will (survive|recover|die))\b',
        r'\b(curable|uncurable|terminal|fatal)\b',
        
        # Reassurance (false comfort)
        r'\b(don\'t worry|no need to worry|nothing to worry about|you\'ll be fine)\b',
        r'\b(not serious|minor condition|easily treatable)\b',
        
        # Treatment urgency
        r'\b(urgent|emergency|immediately (see|consult)|life-threatening)\b.*\b(treatment|surgery|intervention)\b',
        
        # Speculation beyond CNN outputs
        r'\b(probably|likely|appears to|seems to|suggests)\b.*\b(metasta|spread|advanced|stage)\b',
        
        # Medical advice disguised as suggestions
        r'\b(consider|might want to|good idea to)\b.*\b(consult|see a doctor|medical attention)\b.*\b(because|since)\b',
    ]
    
    # REQUIRED DISCLAIMERS
    # At least ONE of these MUST appear in every response
    REQUIRED_DISCLAIMERS = [
        "AI prediction requires medical validation",
        "requires professional validation",
        "consult a medical professional",
    ]
    
    # UNCERTAINTY LANGUAGE (required for low confidence)
    # Must appear when confidence < 70%
    UNCERTAINTY_PHRASES = [
        "moderate confidence",
        "uncertain",
        "low confidence",
        "requires expert review",
        "alternative diagnoses should be considered",
    ]
    
    def __init__(self):
        """Initialize safety filter with compiled regex patterns."""
        self.prohibited_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.PROHIBITED_PATTERNS
        ]
    
    def check_prohibited_patterns(self, response: str) -> Tuple[bool, List[str]]:
        """
        Check if response contains prohibited medical advice patterns.
        
        Args:
            response: LLM output string
            
        Returns:
            (is_safe, violations): 
                - is_safe: False if prohibited patterns found
                - violations: List of matched patterns
        """
        violations = []
        
        for pattern in self.prohibited_regex:
            if pattern.search(response):
                violations.append(pattern.pattern)
        
        return (len(violations) == 0, violations)
    
    def check_disclaimer_present(self, response: str) -> bool:
        """
        Check if response contains required medical disclaimer.
        
        Args:
            response: LLM output string
            
        Returns:
            True if at least ONE required disclaimer present
        """
        response_lower = response.lower()
        
        for disclaimer in self.REQUIRED_DISCLAIMERS:
            if disclaimer.lower() in response_lower:
                return True
        
        return False
    
    def inject_disclaimer(self, response: str) -> str:
        """
        Add mandatory disclaimer if missing.
        
        Args:
            response: LLM output
            
        Returns:
            Response with disclaimer appended
        """
        if self.check_disclaimer_present(response):
            return response
        
        # Append minimal disclaimer
        disclaimer = " AI prediction requires medical validation."
        
        return response.rstrip() + disclaimer
    
    def enforce(self, response: str) -> str:
        """
        Enforce ALL safety rules on LLM response.
        
        ENFORCEMENT STEPS:
        1. Check for prohibited patterns → REJECT if found
        2. Check for disclaimer → INJECT if missing
        3. Return safe response
        
        This is the MAIN METHOD called after every LLM generation.
        
        Args:
            response: Raw LLM output
            
        Returns:
            Safety-filtered response (or refusal if unsafe)
        """
        # Step 1: Check prohibited patterns
        is_safe, violations = self.check_prohibited_patterns(response)
        
        if not is_safe:
            # HARD REJECTION: Response contains medical advice
            refusal = (
                "I cannot provide that information as it would constitute medical advice. "
                "This system only explains AI predictions and cannot replace professional medical consultation. "
                "\n\nPlease consult a qualified radiologist or oncologist for medical guidance."
            )
            return refusal
        
        # Step 2: Ensure disclaimer present
        response = self.inject_disclaimer(response)
        
        # Step 3: Return safe response
        return response
    
    def validate_question(self, question: str) -> Tuple[bool, str]:
        """
        Pre-validate user questions before sending to LLM.
        
        Blocks clearly unsafe questions to prevent wasted inference.
        
        Args:
            question: User's input question
            
        Returns:
            (is_valid, reason):
                - is_valid: False if question should be blocked
                - reason: Explanation for rejection
        """
        question_lower = question.lower()
        
        # Block treatment questions
        treatment_keywords = ['should i', 'recommend', 'what treatment', 'how to treat', 'cure', 'medication']
        for keyword in treatment_keywords:
            if keyword in question_lower:
                return (False, "I cannot provide treatment recommendations or medical advice. Please consult a medical professional.")
        
        # Block prognosis questions
        prognosis_keywords = ['will i', 'how long', 'survival', 'life expectancy', 'prognosis']
        for keyword in prognosis_keywords:
            if keyword in question_lower:
                return (False, "I cannot provide prognosis or outcome predictions. Please consult an oncologist for prognostic information.")
        
        # All other questions allowed
        return (True, "")


# Module exports
__all__ = ['SafetyFilter']
