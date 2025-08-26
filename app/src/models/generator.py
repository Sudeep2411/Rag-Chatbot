from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
from typing import Optional
from app.config import MODEL_ID, HF_TOKEN, USE_OLLAMA, OLLAMA_MODEL, OLLAMA_BASE_URL
import torch
import requests
import json
from app.src.utils.logger import get_logger

logger = get_logger("Generator")

class TextGenerator:
    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or MODEL_ID
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_ollama = USE_OLLAMA
        self.ollama_model = OLLAMA_MODEL
        self.ollama_base_url = OLLAMA_BASE_URL
        
        if not self.use_ollama:
            # Hugging Face initialization
            try:
                logger.info(f"Loading Hugging Face model: {self.model_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, 
                    token=HF_TOKEN,
                    trust_remote_code=True
                )
                
                # Set padding token if not exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_id, 
                        token=HF_TOKEN,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.task = "text2text-generation"
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id, 
                        token=HF_TOKEN,
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.task = "text-generation"
                
                # Move model to device
                self.model.to(self.device)
                
                self.pipe = pipeline(
                    self.task, 
                    model=self.model, 
                    tokenizer=self.tokenizer, 
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Model loaded successfully. Task: {self.task}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.model_id}: {e}")
                raise RuntimeError(f"Failed to load model {self.model_id}: {e}")
        else:
            # Ollama initialization
            logger.info(f"Using Ollama model: {self.ollama_model}")
            try:
                # Test connection to Ollama
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama server not responding: {response.text}")
                
                # Check if model exists
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.ollama_model not in model_names:
                    logger.warning(f"Model {self.ollama_model} not found. Available: {model_names}")
                    logger.info(f"Attempting to pull model {self.ollama_model}...")
                    
                    # Try to pull the model
                    try:
                        pull_response = requests.post(
                            f"{self.ollama_base_url}/api/pull",
                            json={"name": self.ollama_model},
                            timeout=300  # 5 minutes timeout for pulling
                        )
                        if pull_response.status_code != 200:
                            raise RuntimeError(f"Failed to pull model: {pull_response.text}")
                        logger.info(f"Successfully pulled model {self.ollama_model}")
                    except Exception as pull_error:
                        raise RuntimeError(f"Model {self.ollama_model} not found and couldn't be pulled: {pull_error}")
                
                logger.info("Ollama connection successful")
                
            except requests.exceptions.ConnectionError:
                raise RuntimeError("Ollama server not running. Please start Ollama first with 'ollama serve'.")
            except Exception as e:
                raise RuntimeError(f"Ollama initialization failed: {e}")

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9):
        logger.info(f"Generating response for prompt (length: {len(prompt)})")
        if self.use_ollama:
            return self._generate_ollama(prompt, max_new_tokens, temperature, top_p)
        else:
            return self._generate_hf(prompt, max_new_tokens, temperature, top_p)
    
    def _generate_hf(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        try:
            if self.task == "text2text-generation":
                out = self.pipe(
                    prompt, 
                    max_new_tokens=max_new_tokens, 
                    temperature=temperature, 
                    top_p=top_p,
                    do_sample=True
                )[0]["generated_text"]
            else:
                out = self.pipe(
                    prompt, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    temperature=temperature, 
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )[0]["generated_text"]
            
            logger.debug(f"Generated response: {out[:100]}...")
            return out
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_ollama(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens
            },
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate", 
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get("response", "No response generated")
            logger.debug(f"Ollama response: {generated_text[:100]}...")
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"Error generating response: {str(e)}"
