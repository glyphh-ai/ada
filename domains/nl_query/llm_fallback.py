"""
LLM Fallback for Natural Language Query Translation.

Optional component that uses a small LLM (Phi-3.5-mini-instruct) to translate
natural language queries when rules-based matching has low confidence.

Only instantiated when ENABLE_LLM_FALLBACK=true and transformers is available.
"""

import logging
import json
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMFallback:
    """
    LLM-based query translation fallback.
    
    Uses Phi-3.5-mini-instruct (or configured model) to translate
    natural language queries to structured Glyphh queries.
    
    Lazy-loads the model on first use to minimize startup time.
    """
    
    # Prompt template for query translation
    TRANSLATION_PROMPT = """You are a query translator for a semantic knowledge system called Glyphh.

Your task is to translate natural language queries into structured JSON queries.

Available operations:
- similarity_search: Find similar concepts. Parameters: query (string), top_k (int, default 10)
- fact_tree: Verify and explain a claim. Parameters: claim (string), max_depth (int, default 3)
- temporal_predict: Predict future states. Parameters: current_state (list of strings), steps_ahead (int, default 1)
- list: List all concepts. Parameters: limit (int, default 100)
- count: Count concepts. No parameters.
- compare: Compare two concepts. Parameters: query (string with both concepts)

Respond ONLY with valid JSON. No explanation, no markdown, just JSON.

Example inputs and outputs:
- "find similar to machine learning" -> {"operation": "similarity_search", "query": "machine learning", "top_k": 10}
- "verify that neural networks are used in AI" -> {"operation": "fact_tree", "claim": "neural networks are used in AI", "max_depth": 3}
- "predict what comes after data collection" -> {"operation": "temporal_predict", "current_state": ["data collection"], "steps_ahead": 1}
- "how many concepts are there" -> {"operation": "count"}
- "list all items" -> {"operation": "list", "limit": 100}

Now translate this query:
{query}

JSON response:"""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        device: str = "auto",
    ):
        """
        Initialize the LLM Fallback.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._available = None
    
    def is_available(self) -> bool:
        """
        Check if LLM fallback is available.
        
        Returns:
            True if transformers is installed and model can be loaded
        """
        if self._available is not None:
            return self._available
        
        try:
            import transformers
            self._available = True
            logger.info("LLM fallback available (transformers installed)")
        except ImportError:
            self._available = False
            logger.info("LLM fallback not available (transformers not installed)")
        
        return self._available
    
    def _load_model(self) -> bool:
        """
        Lazy-load the model and tokenizer.
        
        Returns:
            True if model loaded successfully
        """
        if self._model is not None:
            return True
        
        if not self.is_available():
            return False
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading LLM model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Load model with appropriate settings
            if device == "cuda":
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
                self._model = self._model.to(device)
            
            logger.info(f"LLM model loaded on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            self._available = False
            return False
    
    async def translate_query(
        self,
        query: str,
        namespace: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Translate a natural language query to structured format.
        
        Args:
            query: Natural language query
            namespace: Target namespace (for context)
            
        Returns:
            Structured query dict or None if translation fails
        """
        if not self._load_model():
            return None
        
        try:
            # Build prompt
            prompt = self.TRANSLATION_PROMPT.format(query=query)
            
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            # Move to model device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            import torch
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode
            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            # Parse JSON from response
            response = response.strip()
            
            # Try to extract JSON if wrapped in other text
            if not response.startswith("{"):
                # Look for JSON in the response
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    response = response[start:end]
            
            result = json.loads(response)
            
            logger.info(f"LLM translated query: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM response not valid JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return None
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Try to free GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("LLM model unloaded")
