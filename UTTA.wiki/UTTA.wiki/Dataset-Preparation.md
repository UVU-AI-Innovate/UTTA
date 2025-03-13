# Dataset Preparation Tutorial

This tutorial will guide you through preparing datasets for LLM fine-tuning. You'll learn how to create, clean, format, and validate your data through practical examples.

## Prerequisites

```bash
# Install required packages
pip install pandas numpy nltk scikit-learn tiktoken
pip install datasets transformers

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Step 1: Creating Your Dataset

### Option 1: Using Existing Data

```python
# example_dataset.py
from datasets import load_dataset

def load_existing_dataset():
    # Load from Hugging Face Hub
    dataset = load_dataset("squad")  # Example using SQuAD
    
    # Convert to pandas for easier processing
    df = dataset['train'].to_pandas()
    
    # Basic formatting
    formatted_data = {
        'question': df['question'],
        'answer': df['answers'].apply(lambda x: x['text'][0])
    }
    
    return pd.DataFrame(formatted_data)

if __name__ == "__main__":
    df = load_existing_dataset()
    print(f"Loaded {len(df)} examples")
    print("\nSample data:")
    print(df.head())
```

### Option 2: Creating Custom Data

```python
# custom_dataset.py
import pandas as pd
from typing import List, Dict

def create_qa_examples() -> List[Dict]:
    return [
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen."
        },
        {
            "question": "Explain the water cycle.",
            "answer": "The water cycle is the continuous movement of water on Earth through evaporation, condensation, precipitation, and collection."
        }
    ]

def create_custom_dataset(examples: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(examples)

if __name__ == "__main__":
    examples = create_qa_examples()
    df = create_custom_dataset(examples)
    df.to_csv('raw_dataset.csv', index=False)
```

## Step 2: Data Cleaning

Create `data_cleaner.py`:

```python
import pandas as pd
import re
import nltk
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.min_length = 20
        self.max_length = 500
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Basic normalization
        text = text.strip()
        
        return text
    
    def has_complete_sentences(self, text: str) -> bool:
        """Check if text contains complete sentences"""
        sentences = nltk.sent_tokenize(text)
        return len(sentences) >= 1
    
    def calculate_quality_score(self, text: str) -> float:
        """Calculate text quality score"""
        # Length score
        length_score = min(1.0, len(text) / self.max_length)
        
        # Sentence structure score
        sentences = nltk.sent_tokenize(text)
        sentence_score = min(1.0, len(sentences) / 3)
        
        # Vocabulary diversity score
        words = text.lower().split()
        diversity_score = len(set(words)) / len(words) if words else 0
        
        # Combined score
        return (length_score + sentence_score + diversity_score) / 3
    
    def clean_dataset(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Clean entire dataset"""
        logger.info("Starting dataset cleaning...")
        initial_size = len(df)
        
        # Clean text
        for col in ['question', 'answer']:
            df[col] = df[col].apply(self.clean_text)
        
        # Remove empty entries
        df = df.dropna()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Filter by length
        df = df[df['answer'].str.len().between(self.min_length, self.max_length)]
        
        # Check for complete sentences
        df = df[df['answer'].apply(self.has_complete_sentences)]
        
        # Calculate quality scores
        df['quality_score'] = df['answer'].apply(self.calculate_quality_score)
        df = df[df['quality_score'] > threshold]
        
        # Log results
        final_size = len(df)
        logger.info(f"Cleaning complete: {initial_size - final_size} examples removed")
        logger.info(f"Final dataset size: {final_size}")
        
        return df

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('raw_dataset.csv')
    cleaner = DataCleaner()
    clean_df = cleaner.clean_dataset(df)
    clean_df.to_csv('clean_dataset.csv', index=False)
```

## Step 3: Format for Different Methods

Create `data_formatter.py`:

```python
import json
import tiktoken
from pathlib import Path
from typing import List, Dict
import logging
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFormatter:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def format_for_openai(self, df: pd.DataFrame, output_file: str):
        """Format data for OpenAI fine-tuning"""
        formatted_data = []
        for _, row in df.iterrows():
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": row['question']},
                    {"role": "assistant", "content": row['answer']}
                ]
            })
        
        # Save as JSONL
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved OpenAI format data to {output_path}")
        return output_path
    
    def format_for_huggingface(self, df: pd.DataFrame) -> Dataset:
        """Format data for HuggingFace fine-tuning"""
        formatted_data = {
            'instruction': ["Answer the following question:"] * len(df),
            'input': df['question'].tolist(),
            'output': df['answer'].tolist()
        }
        
        dataset = Dataset.from_dict(formatted_data)
        logger.info(f"Created HuggingFace dataset with {len(dataset)} examples")
        return dataset
    
    def format_for_dspy(self, df: pd.DataFrame) -> List[Dict]:
        """Format data for DSPy optimization"""
        formatted_data = []
        for _, row in df.iterrows():
            formatted_data.append({
                'question': row['question'],
                'answer': row['answer']
            })
        
        logger.info(f"Created DSPy format data with {len(formatted_data)} examples")
        return formatted_data

def main():
    # Load clean data
    df = pd.read_csv('clean_dataset.csv')
    formatter = DataFormatter()
    
    # Format for each method
    formatter.format_for_openai(df, 'openai_data.jsonl')
    hf_dataset = formatter.format_for_huggingface(df)
    dspy_data = formatter.format_for_dspy(df)
    
    # Save HuggingFace dataset
    hf_dataset.save_to_disk('hf_dataset')
    
    # Save DSPy data
    with open('dspy_data.json', 'w') as f:
        json.dump(dspy_data, f, indent=2)

if __name__ == "__main__":
    main()
```

## Step 4: Validation

Create `data_validator.py`:

```python
import json
import tiktoken
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def validate_openai_format(self, file_path: str) -> Dict[str, Any]:
        """Validate OpenAI format data"""
        stats = {'total': 0, 'valid': 0, 'invalid': 0, 'errors': []}
        
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    stats['total'] += 1
                    try:
                        # Parse JSON
                        data = json.loads(line)
                        
                        # Check structure
                        if 'messages' not in data:
                            raise ValueError("Missing 'messages' key")
                        
                        # Check messages
                        messages = data['messages']
                        roles = [msg.get('role') for msg in messages]
                        
                        if 'system' not in roles:
                            raise ValueError("Missing system message")
                        if 'user' not in roles:
                            raise ValueError("Missing user message")
                        if 'assistant' not in roles:
                            raise ValueError("Missing assistant message")
                        
                        # Check token count
                        total_tokens = sum(len(self.encoding.encode(msg['content']))
                                         for msg in messages)
                        if total_tokens > 4096:
                            raise ValueError("Token limit exceeded")
                        
                        stats['valid'] += 1
                    
                    except Exception as e:
                        stats['invalid'] += 1
                        stats['errors'].append(f"Line {i}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None
        
        return stats
    
    def validate_huggingface_format(self, dataset_path: str) -> Dict[str, Any]:
        """Validate HuggingFace format data"""
        try:
            dataset = Dataset.load_from_disk(dataset_path)
            stats = {
                'total': len(dataset),
                'valid': 0,
                'invalid': 0,
                'errors': []
            }
            
            # Check features
            required_features = ['instruction', 'input', 'output']
            if not all(feat in dataset.features for feat in required_features):
                missing = [feat for feat in required_features if feat not in dataset.features]
                stats['errors'].append(f"Missing features: {missing}")
                return stats
            
            # Validate examples
            for i, example in enumerate(dataset):
                try:
                    # Check types
                    if not all(isinstance(example[feat], str) for feat in required_features):
                        raise ValueError("Invalid data type")
                    
                    # Check for empty strings
                    if any(not example[feat].strip() for feat in required_features):
                        raise ValueError("Empty string found")
                    
                    stats['valid'] += 1
                
                except Exception as e:
                    stats['invalid'] += 1
                    stats['errors'].append(f"Example {i}: {str(e)}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None
    
    def validate_dspy_format(self, file_path: str) -> Dict[str, Any]:
        """Validate DSPy format data"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            stats = {
                'total': len(data),
                'valid': 0,
                'invalid': 0,
                'errors': []
            }
            
            for i, example in enumerate(data):
                try:
                    # Check required keys
                    if not all(key in example for key in ['question', 'answer']):
                        raise ValueError("Missing required keys")
                    
                    # Check types
                    if not all(isinstance(example[key], str) for key in ['question', 'answer']):
                        raise ValueError("Invalid data type")
                    
                    # Check content
                    if not all(example[key].strip() for key in ['question', 'answer']):
                        raise ValueError("Empty string found")
                    
                    stats['valid'] += 1
                
                except Exception as e:
                    stats['invalid'] += 1
                    stats['errors'].append(f"Example {i}: {str(e)}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return None

def main():
    validator = DataValidator()
    
    # Validate OpenAI format
    openai_stats = validator.validate_openai_format('openai_data.jsonl')
    logger.info("OpenAI format validation:")
    logger.info(json.dumps(openai_stats, indent=2))
    
    # Validate HuggingFace format
    hf_stats = validator.validate_huggingface_format('hf_dataset')
    logger.info("HuggingFace format validation:")
    logger.info(json.dumps(hf_stats, indent=2))
    
    # Validate DSPy format
    dspy_stats = validator.validate_dspy_format('dspy_data.json')
    logger.info("DSPy format validation:")
    logger.info(json.dumps(dspy_stats, indent=2))

if __name__ == "__main__":
    main()
```

## Complete Pipeline Example

Create `prepare_dataset.py`:

```python
from pathlib import Path
import logging
from data_cleaner import DataCleaner
from data_formatter import DataFormatter
from data_validator import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPipeline:
    def __init__(self):
        self.cleaner = DataCleaner()
        self.formatter = DataFormatter()
        self.validator = DataValidator()
    
    def run_pipeline(self, input_file: str, output_dir: str):
        """Run complete dataset preparation pipeline"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load and clean data
        logger.info("Loading and cleaning data...")
        df = pd.read_csv(input_file)
        clean_df = self.cleaner.clean_dataset(df)
        
        # Step 2: Format for each method
        logger.info("Formatting data...")
        openai_path = output_path / 'openai_data.jsonl'
        self.formatter.format_for_openai(clean_df, openai_path)
        
        hf_dataset = self.formatter.format_for_huggingface(clean_df)
        hf_dataset.save_to_disk(output_path / 'hf_dataset')
        
        dspy_data = self.formatter.format_for_dspy(clean_df)
        with open(output_path / 'dspy_data.json', 'w') as f:
            json.dump(dspy_data, f, indent=2)
        
        # Step 3: Validate all formats
        logger.info("Validating data...")
        validation_results = {
            'openai': self.validator.validate_openai_format(str(openai_path)),
            'huggingface': self.validator.validate_huggingface_format(str(output_path / 'hf_dataset')),
            'dspy': self.validator.validate_dspy_format(str(output_path / 'dspy_data.json'))
        }
        
        # Save validation results
        with open(output_path / 'validation_results.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info("Pipeline complete! Results saved to validation_results.json")
        return validation_results

if __name__ == "__main__":
    pipeline = DatasetPipeline()
    results = pipeline.run_pipeline('raw_dataset.csv', 'processed_data')
```

## Usage Example

```bash
# Create and process dataset
python custom_dataset.py
python prepare_dataset.py

# Check results
cat processed_data/validation_results.json
```

## Best Practices

1. **Data Quality**:
   - Use diverse examples
   - Ensure consistent formatting
   - Remove duplicates
   - Validate content quality

2. **Format-Specific**:
   - OpenAI: Follow JSONL format
   - HuggingFace: Use Dataset format
   - DSPy: Keep simple dictionary structure

3. **Validation**:
   - Check data types
   - Verify token limits
   - Validate format requirements
   - Log all errors

4. **Error Handling**:
   - Implement proper logging
   - Use try-except blocks
   - Save validation results
   - Track statistics

## Next Steps

1. Choose your fine-tuning method:
   - [DSPy Tutorial](DSPy-Tutorial.md)
   - [OpenAI Tutorial](OpenAI-Tutorial.md)
   - [HuggingFace Tutorial](HuggingFace-Tutorial.md)

2. Prepare your dataset:
   - Collect relevant data
   - Run the preparation pipeline
   - Validate the results

3. Start fine-tuning:
   - Follow method-specific tutorial
   - Use prepared dataset
   - Monitor training progress

## Resources

- [OpenAI Data Preparation Guide](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets)
- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [NLTK Documentation](https://www.nltk.org/) 