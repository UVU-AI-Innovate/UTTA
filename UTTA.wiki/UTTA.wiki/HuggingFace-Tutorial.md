# HuggingFace LoRA Fine-Tuning Tutorial

This tutorial will guide you through fine-tuning a large language model using HuggingFace's PEFT library with LoRA (Low-Rank Adaptation). You'll learn how to prepare your data, implement LoRA fine-tuning, and use your fine-tuned model.

## Prerequisites

- Completed [Environment Setup](Environment-Setup.md)
- GPU with 8GB+ VRAM
- Basic PyTorch knowledge
- Prepared dataset (see [Dataset Preparation](Dataset-Preparation.md))

## Step 1: Setup

```bash
# Install required packages
pip install torch transformers datasets peft evaluate accelerate bitsandbytes
pip install tensorboard  # for training monitoring

# Verify installations
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

Create `setup_utils.py`:
```python
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSetup:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_base_model(self):
        """Load and prepare base model for LoRA fine-tuning"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load model in 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        return model
    
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def create_lora_config(self):
        """Create LoRA configuration"""
        return LoraConfig(
            r=16,  # rank
            lora_alpha=32,  # scaling factor
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def prepare_model(self):
        """Prepare complete model for training"""
        # Load components
        base_model = self.load_base_model()
        tokenizer = self.load_tokenizer()
        lora_config = self.create_lora_config()
        
        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model, tokenizer

if __name__ == "__main__":
    setup = ModelSetup()
    model, tokenizer = setup.prepare_model()
    logger.info("Setup complete!")
```

## Step 2: Data Preparation

Create `prepare_data.py`:
```python
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_dataset(self, dataset_path: str) -> Dataset:
        """Load dataset from disk"""
        return load_from_disk(dataset_path)
    
    def format_prompt(self, example: Dict) -> str:
        """Format example into prompt"""
        return f"""### Question: {example['input']}

### Answer: {example['output']}

"""
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize examples"""
        prompts = [self.format_prompt(ex) for ex in examples]
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Prepare labels (same as input_ids for causal LM)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_dataset(self, dataset_path: str):
        """Prepare complete dataset for training"""
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def create_dataloaders(self, dataset: Dataset, batch_size: int = 4):
        """Create train and validation dataloaders"""
        # Split dataset
        dataset = dataset.train_test_split(test_size=0.1)
        
        # Create dataloaders
        train_dataloader = DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            dataset["test"],
            batch_size=batch_size
        )
        
        return train_dataloader, val_dataloader

if __name__ == "__main__":
    from setup_utils import ModelSetup
    
    # Setup
    setup = ModelSetup()
    _, tokenizer = setup.prepare_model()
    
    # Prepare data
    preparator = DataPreparator(tokenizer)
    dataset = preparator.prepare_dataset("processed_data/hf_dataset")
    train_dataloader, val_dataloader = preparator.create_dataloaders(dataset)
    
    logger.info("Data preparation complete!")
```

## Step 3: Training Process

Create `train.py`:
```python
import torch
from transformers import Trainer, TrainingArguments
from peft import PeftModel
import logging
from pathlib import Path
import json
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoRATrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        val_dataset,
        output_dir: str = "results"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
    
    def create_training_args(self):
        """Create training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=50,
            warmup_steps=50,
            weight_decay=0.01,
            report_to="tensorboard"
        )
    
    def train(self):
        """Run training process"""
        # Create training arguments
        training_args = self.create_training_args()
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.output_dir / "final_model"))
        
        # Save training stats
        stats = {
            "train_loss": trainer.state.log_history[-1]["train_loss"],
            "eval_loss": trainer.state.log_history[-1]["eval_loss"]
        }
        
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Training complete!")
        return stats

if __name__ == "__main__":
    from setup_utils import ModelSetup
    from prepare_data import DataPreparator
    
    # Setup
    setup = ModelSetup()
    model, tokenizer = setup.prepare_model()
    
    # Prepare data
    preparator = DataPreparator(tokenizer)
    dataset = preparator.prepare_dataset("processed_data/hf_dataset")
    
    # Train
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        val_dataset=dataset["test"]
    )
    stats = trainer.train()
```

## Step 4: Testing and Usage

Create `test_model.py`:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from pathlib import Path
import json
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_path: str, base_model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        logger.info("Loading model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer(self, question: str) -> str:
        """Generate answer for a question"""
        # Format prompt
        prompt = f"### Question: {question}\n\n### Answer:"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                num_return_sequences=1
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part
        answer = answer.split("### Answer:")[-1].strip()
        
        return answer
    
    def run_test_cases(self, test_cases: List[Dict]) -> List[Dict]:
        """Run multiple test cases"""
        results = []
        
        for case in test_cases:
            question = case["question"]
            expected = case["answer"]
            generated = self.generate_answer(question)
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": generated
            })
        
        return results

def main():
    # Test cases
    test_cases = [
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy."
        },
        {
            "question": "Explain Newton's first law.",
            "answer": "Newton's first law states that an object remains at rest or in motion unless acted upon by an external force."
        }
    ]
    
    # Run tests
    tester = ModelTester("results/final_model")
    results = tester.run_test_cases(test_cases)
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("Test results saved to test_results.json")

if __name__ == "__main__":
    main()
```

## Running the Tutorial

1. **Setup and Verification**:
   ```bash
   python setup_utils.py
   ```

2. **Prepare Data**:
   ```bash
   python prepare_data.py
   ```

3. **Run Training**:
   ```bash
   python train.py
   ```

4. **Test Model**:
   ```bash
   python test_model.py
   ```

## Cost and Resource Requirements

- **GPU**: 8GB+ VRAM (16GB+ recommended)
- **Storage**: ~20GB for model and datasets
- **RAM**: 16GB+ recommended
- **Training Time**: 2-8 hours depending on dataset size
- **Cost**: Computing resources only (no API costs)

## Best Practices

1. **Data Preparation**:
   - Clean and validate data
   - Use consistent formatting
   - Balance dataset
   - Check token lengths

2. **Training**:
   - Start with small datasets
   - Monitor training loss
   - Use gradient accumulation
   - Save checkpoints

3. **Resource Management**:
   - Use 4-bit quantization
   - Implement gradient checkpointing
   - Monitor GPU memory
   - Use appropriate batch size

4. **Model Usage**:
   - Implement proper error handling
   - Use appropriate generation parameters
   - Cache results when possible
   - Monitor inference time

## Troubleshooting

1. **CUDA Out of Memory**:
   ```python
   # Solutions:
   # 1. Reduce batch size
   training_args = TrainingArguments(
       per_device_train_batch_size=2,  # Reduce from 4
       gradient_accumulation_steps=8    # Increase from 4
   )
   
   # 2. Enable gradient checkpointing
   model.gradient_checkpointing_enable()
   ```

2. **Training Issues**:
   - Monitor loss curves
   - Check learning rate
   - Validate data format
   - Inspect generated outputs

3. **Inference Problems**:
   - Check GPU memory
   - Reduce generation length
   - Validate input format
   - Monitor temperature

## Next Steps

1. **Advanced Topics**:
   - [Hyperparameter Tuning](Hyperparameter-Tuning.md)
   - [Model Evaluation](Evaluation-Metrics.md)
   - [Production Deployment](Model-Deployment.md)

2. **Alternative Methods**:
   - [DSPy Tutorial](DSPy-Tutorial.md)
   - [OpenAI Tutorial](OpenAI-Tutorial.md)

## Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Best Practices Guide](https://huggingface.co/docs/transformers/v4.18.0/en/performance) 