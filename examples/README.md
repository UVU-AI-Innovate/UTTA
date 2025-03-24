# 🎓 Educational Examples: Fine-tuning Methods for LLMs

This directory contains educational examples and assignments designed to teach different approaches to fine-tuning Large Language Models (LLMs) for educational applications. These examples are specifically crafted for students learning about LLM adaptation techniques.

## 📚 Learning Objectives

After working through these examples, students will be able to:
1. Master three distinct fine-tuning approaches:
   - Prompt optimization with DSPy (focusing on prompt engineering and optimization)
   - Cloud-based fine-tuning with OpenAI (understanding API-based model adaptation)
   - Local parameter-efficient fine-tuning with LoRA (learning local model modification)
2. Implement each approach independently using real educational datasets
3. Analyze the unique characteristics of each method:
   - Data requirements and preparation
   - Computational resource needs
   - Cost considerations
   - Performance characteristics
   - Implementation complexity
4. Make informed decisions about which approach best suits different educational scenarios

## 📂 Repository Structure

```
examples/
├── README.md                                 # This file - Overview and instructions
│
├── Approach 1: DSPy Prompt Optimization
│   ├── dspy_example.py                      # Main DSPy implementation
│   ├── teacher_student_dialogues.jsonl      # Training dialogues
│   └── utils.py                             # Helper functions
│
├── Approach 2: OpenAI Fine-tuning
│   ├── openai_finetune.py                   # OpenAI fine-tuning implementation
│   ├── openai_teacher_dialogue_training.jsonl
│   └── openai_teacher_dialogue_validation.jsonl
│
└── Approach 3: HuggingFace LoRA
    ├── huggingface_lora.py                  # LoRA implementation
    ├── small_edu_qa.jsonl                   # Sample QA dataset
    └── openai_edu_qa_training.jsonl         # Additional training data
```

## 🔍 Three Approaches Explained

### 1. DSPy: Prompt Optimization (dspy_example.py)

**Key Features:**
- No model weight changes - only better prompting
- Works with smaller datasets (10-50 dialogue examples)
- Uses Chain-of-Thought prompting for pedagogical reasoning
- API-based approach with immediate results
- Includes evaluation metrics for teaching quality

**Implementation Details:**
```python
class StudentResponse(dspy.Signature):
    """Define the input/output structure for student responses"""
    conversation_history = dspy.InputField()
    teacher_prompt = dspy.InputField()
    student_response = dspy.OutputField()

class StudentSimulationModel(dspy.Module):
    """Implement the student simulation logic"""
    def __init__(self):
        # Initialize with chain-of-thought reasoning
        self.chain = dspy.ChainOfThought(StudentResponse)
```

### 2. OpenAI: Cloud Fine-tuning (openai_finetune.py)

**Key Features:**
- Updates actual model weights through cloud service
- Requires medium-sized datasets (50-100+ dialogue examples)
- Balance of control and convenience
- Handles multi-turn conversations with context preservation

**Implementation Details:**
```python
def prepare_training_data(dialogues):
    """Convert educational dialogues to OpenAI format"""
    return [{
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": dialogue["student"]},
            {"role": "assistant", "content": dialogue["teacher"]}
        ]
    } for dialogue in dialogues]
```

### 3. HuggingFace: Local LoRA (huggingface_lora.py)

**Key Features:**
- Full local control over model and training process
- Uses LoRA for efficient adaptation of model weights
- Works best with larger datasets (100-1000+ dialogue examples)
- Data privacy - all information stays on your machine

**Implementation Details:**
```python
def prepare_model_and_tokenizer():
    """Set up base model with LoRA configuration"""
    model = AutoModelForCausalLM.from_pretrained(
        "base_model",
        load_in_8bit=True,
        device_map="auto"
    )
    
    config = LoraConfig(
        r=8,  # Rank of update matrices
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
```

## 📊 Dataset Formats

### 1. teacher_student_dialogues.jsonl
Multi-turn dialogues for DSPy:
```json
{
  "dialogue": [
    {"role": "teacher", "content": "Can you explain what photosynthesis is?"},
    {"role": "student", "content": "It's how plants make their food, right?"},
    {"role": "teacher", "content": "That's a good start! Can you tell me more?"}
  ]
}
```

### 2. openai_teacher_dialogue_training.jsonl
OpenAI chat completion format:
```json
{
  "messages": [
    {"role": "system", "content": "You are a knowledgeable teacher..."},
    {"role": "user", "content": "What is the water cycle?"},
    {"role": "assistant", "content": "The water cycle is the continuous movement..."}
  ]
}
```

### 3. small_edu_qa.jsonl
Simple QA pairs for LoRA:
```json
{
  "instruction": "Explain the concept of gravity to a middle school student.",
  "output": "Gravity is like an invisible force that pulls objects toward each other..."
}
```

## 📝 Assignment Structure

Each approach should be studied independently in this order:

1. **DSPy Prompt Optimization (40% of learning)**
   - Study and run dspy_example.py
   - Create new educational dialogues
   - Implement enhancements
   - Evaluate performance

2. **OpenAI Fine-tuning (30% of learning)**
   - Prepare domain-specific QA pairs
   - Configure fine-tuning parameters
   - Analyze costs and benefits

3. **HuggingFace LoRA (30% of learning)**
   - Set up the local environment
   - Understand LoRA parameters
   - Implement and evaluate

## 💡 Recommended Learning Path

For each approach:

1. **Study the Implementation**
   - Read through the code thoroughly
   - Understand key components
   - Review documentation

2. **Prepare the Environment**
   - Set up dependencies
   - Configure API keys/settings
   - Verify setup

3. **Run and Experiment**
   - Test with provided data
   - Try different parameters
   - Document results

4. **Implement Enhancements**
   - Add improvements
   - Test with new data
   - Document changes

5. **Evaluate and Analyze**
   - Compare performance
   - Document limitations
   - Suggest improvements

## 📚 Resources

- DSPy Documentation: [https://dspy.ai/](https://dspy.ai/)
- OpenAI Fine-tuning Guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- HuggingFace LoRA Guide: [https://huggingface.co/docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora)

## 🤝 Support

If you encounter issues:
1. Check the documentation in each example file
2. Review the code comments
3. Contact course staff through designated channels

## 📊 Evaluation Criteria

Your work will be evaluated on:

1. **Technical Implementation (40%)**
   - Correct implementation
   - Code quality
   - Error handling

2. **Analysis & Understanding (30%)**
   - Depth of analysis
   - Understanding of tradeoffs
   - Documentation quality

3. **Creativity & Enhancement (20%)**
   - Innovative improvements
   - Novel evaluation metrics
   - Creative solutions

4. **Presentation (10%)**
   - Clear documentation
   - Organized code
   - Professional presentation

Good luck with your learning journey! 🚀
