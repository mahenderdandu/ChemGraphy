"""
LLM wrapper for Helios chemistry question answering
Supports local models (Llama, Mistral) and API models
"""

import torch
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 2048
    temperature: float = 0.3  # Lower for more focused, factual responses
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15  # Higher to prevent repetition
    do_sample: bool = True
    num_beams: int = 1


class HeliosLLM:
    """
    LLM wrapper for Helios chemistry assistant
    Supports transformers models and API backends
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        model_type: str = "api",  # "api", "local", "hf"
        device: str = "cpu",
        generation_config: Optional[GenerationConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Args:
            model_name: Model name/path
            model_type: "api" (OpenAI), "local" (transformers), "hf" (HuggingFace Hub)
            device: Device for local models
            generation_config: Generation parameters
            api_key: API key for API models
        """
        self.model_name = model_name
        self.model_type = model_type
        self.device = device
        self.generation_config = generation_config or GenerationConfig()
        self.api_key = api_key
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initializing {model_type} LLM: {model_name}")
        
        if model_type == "local" or model_type == "hf":
            self._load_local_model()
        elif model_type == "api":
            self._init_api_client()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _load_local_model(self):
        """Load local transformers model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            logger.info(f"Loading model from {self.model_name}")
            
            # Check if this is a model that needs quantization (7B+ models)
            use_4bit = "7b" in self.model_name.lower() or "mistral" in self.model_name.lower()
            
            if use_4bit:
                logger.info("Using 4-bit quantization for large model")
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
                    self.pipe = None  # Use direct model inference
                    
                except Exception as e:
                    logger.warning(f"4-bit quantization failed: {e}, falling back to standard loading")
                    use_4bit = False
            
            if not use_4bit:
                from transformers import pipeline
                
                # Use pipeline for smaller models
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype=torch.float32,
                    device="cpu",
                    trust_remote_code=True,
                    model_kwargs={"low_cpu_mem_usage": True}
                )
                
                self.tokenizer = self.pipe.tokenizer
                self.model = self.pipe.model
            
            self.model.eval()
            logger.info(f"Model loaded successfully")
            
        except ImportError:
            logger.error("transformers library not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _init_api_client(self):
        """Initialize API client"""
        try:
            import openai
            
            if self.api_key:
                openai.api_key = self.api_key
            
            logger.info(f"API client initialized for {self.model_name}")
            
        except ImportError:
            logger.warning("openai library not installed. API calls will fail.")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_new_tokens: Override max tokens
            temperature: Override temperature
            top_p: Override top_p
            stop_sequences: Stop generation at these sequences
        
        Returns:
            Generated text
        """
        config = self.generation_config
        temperature = temperature if temperature is not None else config.temperature
        top_p = top_p if top_p is not None else config.top_p
        max_new_tokens = max_new_tokens if max_new_tokens is not None else config.max_new_tokens
        
        if self.model_type == "api":
            return self._generate_api(prompt, system_prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_local(prompt, system_prompt, max_new_tokens, temperature, top_p, stop_sequences)
    
    def _generate_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Generate using API"""
        try:
            import openai
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return f"Error: {str(e)}"
    
    def _generate_local(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> str:
        """Generate using local model"""
        try:
            # If prompt already has chat template formatting, use it directly
            # Otherwise, format with system message
            if "<|system|>" in prompt or "<|user|>" in prompt:
                full_prompt = prompt
            elif system_prompt:
                full_prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            else:
                full_prompt = prompt
            
            # Check if using Mistral format
            if "mistral" in self.model_name.lower():
                # Mistral uses [INST] format
                if system_prompt:
                    full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
                else:
                    full_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Generate using direct model or pipeline
            if self.pipe is None:
                # Direct model inference (for quantized models)
                inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if temperature > 0 else 1.0,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        repetition_penalty=1.15,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            else:
                # Pipeline inference
                outputs = self.pipe(
                    full_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    repetition_penalty=1.2,
                    return_full_text=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                generated_text = outputs[0]['generated_text']
            
            # Clean up chat template artifacts
            generated_text = generated_text.replace("</s>", "").strip()
            
            # Apply stop sequences
            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text[:generated_text.index(stop_seq)]
            
            # Additional cleanup - stop at common chat artifacts
            for artifact in ["<|user|>", "<|system|>", "\n\nQuestion:", "\n\nContext:"]:
                if artifact in generated_text:
                    generated_text = generated_text[:generated_text.index(artifact)]
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return f"Error: {str(e)}"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Chat-style generation with message history
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Assistant response
        """
        # Convert messages to prompt format
        system_prompt = None
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                conversation.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation.append(f"Assistant: {msg['content']}")
        
        prompt = "\n".join(conversation) + "\nAssistant:"
        
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )


def create_chemistry_llm(
    model_name: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    device: str = "cpu"
) -> HeliosLLM:
    """
    Create LLM instance with sensible defaults
    
    Args:
        model_name: Model to use (default: gpt-3.5-turbo for API, TinyLlama for local)
        use_api: Use OpenAI API instead of local model
        api_key: OpenAI API key
        device: Device for local models
    
    Returns:
        HeliosLLM instance
    """
    if use_api:
        model_name = model_name or "gpt-3.5-turbo"
        return HeliosLLM(
            model_name=model_name,
            model_type="api",
            api_key=api_key
        )
    else:
        # Default to small model for CPU inference
        model_name = model_name or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        return HeliosLLM(
            model_name=model_name,
            model_type="hf",
            device=device
        )


class ChemistryLLM:
    """
    Chemistry-specific LLM for finetuning
    
    Features:
    - Load base model (Mistral/Llama/Qwen)
    - LoRA configuration for efficient finetuning
    - Prompt templating
    - Generation with context
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        use_lora: bool = True,
        lora_config: Optional[Dict] = None,
    ):
        self.model_name = model_name
        self.device_map = device_map
        
        # TODO: Load model and tokenizer
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # from peft import LoraConfig, get_peft_model
        
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     load_in_4bit=load_in_4bit,
        #     device_map=device_map,
        #     torch_dtype=torch.float16,
        # )
        
        # if use_lora:
        #     lora_config = lora_config or self._default_lora_config()
        #     self.model = get_peft_model(self.model, LoraConfig(**lora_config))
        
        logger.info(f"Initialized HeliosLLM with {model_name}")
    
    def _default_lora_config(self) -> Dict:
        """Default LoRA configuration"""
        return {
            'r': 64,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': [
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj'
            ],
            'bias': 'none',
            'task_type': 'CAUSAL_LM',
        }
    
    def format_prompt(
        self,
        query: str,
        context: Optional[List[str]] = None,
        molecule_info: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format prompt for the model
        
        Args:
            query: User query
            context: Retrieved context passages
            molecule_info: Molecule information (SMILES, properties, etc.)
            system_prompt: System instruction
        
        Returns:
            Formatted prompt string
        """
        # TODO: Implement prompt templating
        # Use model-specific format (e.g., Mistral's [INST] tags)
        
        if system_prompt is None:
            system_prompt = (
                "You are Helios, an expert chemistry AI assistant. "
                "Provide accurate, grounded responses with citations. "
                "Admit when you don't know something."
            )
        
        prompt_parts = [f"System: {system_prompt}\n"]
        
        if context:
            prompt_parts.append("Context:")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"[{i}] {ctx}")
            prompt_parts.append("")
        
        if molecule_info:
            prompt_parts.append("Molecule Information:")
            for key, value in molecule_info.items():
                prompt_parts.append(f"- {key}: {value}")
            prompt_parts.append("")
        
        prompt_parts.append(f"Query: {query}\n")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: Input prompt
            generation_config: Generation parameters
        
        Returns:
            Generated text
        """
        # TODO: Implement generation
        # config = generation_config or GenerationConfig()
        
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # outputs = self.model.generate(
        #     **inputs,
        #     max_new_tokens=config.max_new_tokens,
        #     temperature=config.temperature,
        #     top_p=config.top_p,
        #     top_k=config.top_k,
        #     repetition_penalty=config.repetition_penalty,
        #     do_sample=config.do_sample,
        #     num_beams=config.num_beams,
        # )
        
        # response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # return response
        
        raise NotImplementedError("Generation not implemented")
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        molecule_info: Optional[Dict] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate response with RAG context
        
        Args:
            query: User query
            context: Retrieved passages
            molecule_info: Molecule information
            generation_config: Generation parameters
        
        Returns:
            Dictionary with response and metadata
        """
        # Format prompt
        prompt = self.format_prompt(query, context, molecule_info)
        
        # Generate
        response = self.generate(prompt, generation_config)
        
        return {
            'response': response,
            'prompt': prompt,
            'context': context,
            'molecule_info': molecule_info,
        }
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer"""
        # TODO: Implement
        logger.info(f"Saving model to {output_dir}")
        # self.model.save_pretrained(output_dir)
        # self.tokenizer.save_pretrained(output_dir)
        pass
    
    def load_model(self, model_path: str):
        """Load finetuned model"""
        # TODO: Implement
        logger.info(f"Loading model from {model_path}")
        pass


def load_helios_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> HeliosLLM:
    """
    Load Helios LLM
    
    Args:
        model_name: Base model name
        checkpoint_path: Path to finetuned checkpoint
        **kwargs: Additional arguments for HeliosLLM
    
    Returns:
        HeliosLLM instance
    """
    llm = HeliosLLM(model_name, **kwargs)
    
    if checkpoint_path:
        llm.load_model(checkpoint_path)
    
    return llm


# TODO: Add instruction finetuning script
# TODO: Add preference tuning (DPO/PPO)
# TODO: Add evaluation metrics
# TODO: Add batch inference utilities
