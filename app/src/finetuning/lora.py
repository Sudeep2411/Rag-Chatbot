from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
from typing import Dict, List, Optional
import os
from app.src.utils.logger import get_logger

logger = get_logger("LoRAFinetuning")

class LoRAFinetuner:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_config = None
        logger.info(f"Initializing LoRA finetuner with model: {model_name}")
    
    def setup_model(self):
        """Setup the model and tokenizer for fine-tuning"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Setup LoRA configuration
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            self.model = get_peft_model(self.model, self.peft_config)
            self.model.print_trainable_parameters()
            
            logger.info("Model setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up model: {e}")
            return False
    
    def prepare_training_data(self, conversations: List[Dict]):
        """Prepare conversation data for training"""
        try:
            texts = []
            for conv in conversations:
                # Format as: "User: {query}\nAssistant: {response}</s>"
                text = f"User: {conv['query']}\nAssistant: {conv['response']}{self.tokenizer.eos_token}"
                texts.append(text)
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create dataset
            dataset = Dataset.from_dict({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].clone()  # For causal LM, labels are same as input_ids
            })
            
            logger.info(f"Prepared training data with {len(conversations)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    def finetune(self, dataset, output_dir: str = "./lora_finetuned", 
                 num_train_epochs: int = 3, learning_rate: float = 2e-4):
        """Fine-tune the model using LoRA"""
        if not self.model or not self.tokenizer:
            logger.error("Model not setup. Call setup_model() first.")
            return None
        
        try:
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                num_train_epochs=num_train_epochs,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
                warmup_steps=100,
                prediction_loss_only=True,
                fp16=torch.cuda.is_available(),
                optim="adamw_torch",
                report_to=None  # Disable external logging
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            logger.info("Starting fine-tuning...")
            trainer.train()
            
            # Save the model
            trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            return trainer
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return None
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7):
        """Generate text using the fine-tuned model"""
        if not self.model or not self.tokenizer:
            logger.error("Model not setup. Call setup_model() first.")
            return "Model not initialized"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error generating response: {str(e)}"
    
    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info(f"Loaded fine-tuned model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {e}")
            return False
