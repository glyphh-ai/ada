"""
LLM Fallback for Natural Language Query.

Optional LLM-based query translation when rules-based
matching fails. Uses Phi-3.5-mini-instruct by default.
"""

import logging
import re
from typing import Any, Dict, Optional

from domains.nl_query.default_intents import (
    INTENT_SIMILARITY_SEARCH,
    INTENT_FACT_TREE,
    INTENT_TEMPORAL_PREDICTION,
    INTENT_GLYPH_LOOKUP,
    INTENT_EDGE_QUERY,
)
from domains.nl_query.intent_matcher import MatchResult
from infrastructure.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# Prompt template for query translation
TRANSLATION_PROMPT = """You are a query translator. Translate the user's natural language query into a structured query.

Available query types:
1. similarity_search - Find similar concepts (parameters: query)
2. fact_tree - Verify a claim with evidence (parameters: claim)
3. temporal_prediction - Predict future/past states (parameters: state)
4. glyph_lookup - Get a specific glyph by ID (parameters: id)
5. edge_query - Find relationships between concepts (parameters: source, target)

User query: {query}

Respond with ONLY a JSON object in this format:
{{"intent": "<query_type>", "parameters": {{"<param>": "<value>"}}}}

Example responses:
- "find similar to machine learning" -> {{"intent": "similarity_search", "parameters": {{"query": "machine learning"}}}}
- "is it true that Python is popular" -> {{"intent": "fact_tree", "parameters": {{"claim": "Python is popular"}}}}
- "what comes after training" -> {{"intent": "temporal_prediction", "parameters": {{"state": "training"}}}}

JSON response:"""


class LLMFallback:
    """
    LLM-based fallback for query translation.
    
    Uses a small language model (Phi-3.5-mini-instruct) to
    translate natural language queries when rules fail.
    
    Only instantiated when ENABLE_NL_QUERY=true.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize LLMFallback.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("auto", "cpu", "cuda")
        """
        self._model_name = model_name or settings.nl_model
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading NL model: {self._model_name}")
            
            # Determine device
            if self._device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self._device
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device,
                trust_remote_code=True,
            )
            
            self._loaded = True
            logger.info(f"NL model loaded on {device}")
            
        except ImportError as e:
            logger.error(f"transformers not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load NL model: {e}")
            raise
    
    async def translate(self, query: str) -> Optional[MatchResult]:
        """
        Translate a natural language query using LLM.
        
        Args:
            query: Natural language query
            
        Returns:
            MatchResult if translation successful, None otherwise
        """
        try:
            # Lazy load model
            self._load_model()
            
            # Format prompt
            prompt = TRANSLATION_PROMPT.format(query=query)
            
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            # Move to device
            if hasattr(self._model, "device"):
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            
            # Decode
            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Parse JSON response
            result = self._parse_response(response)
            
            if result:
                return MatchResult(
                    intent=result["intent"],
                    confidence=0.8,  # LLM confidence is lower than rules
                    parameters=result.get("parameters", {}),
                    pattern_matched=None,
                )
            
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
        
        return None
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from LLM."""
        import json
        
        # Clean response
        response = response.strip()
        
        # Try to extract JSON
        try:
            # Find JSON object in response
            match = re.search(r'\{[^{}]*\}', response)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        # Try parsing the whole response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        logger.warning(f"Failed to parse LLM response: {response[:100]}")
        return None
    
    def is_available(self) -> bool:
        """Check if LLM fallback is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False
