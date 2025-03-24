# Fine-tuning Methods in Educational AI: A Hands-on Assignment

## Introduction

In this assignment, you will explore three different approaches to improving Large Language Model (LLM) performance for educational applications. You'll work with practical examples that demonstrate prompt optimization, cloud-based fine-tuning, and local fine-tuning with parameter-efficient techniques. Each approach offers unique advantages and tradeoffs that you'll discover through hands-on implementation.

## Learning Objectives

After completing this assignment, you will be able to:
1. Compare and contrast three different fine-tuning approaches:
   - Prompt optimization with DSPy
   - Cloud-based fine-tuning with OpenAI
   - Local parameter-efficient fine-tuning with LoRA
2. Implement and evaluate each approach using real educational datasets
3. Analyze the tradeoffs between:
   - Data requirements
   - Computational resources
   - Cost considerations
   - Model performance
   - Implementation complexity
4. Make informed decisions about which approach best suits different educational scenarios

## Prerequisites

- Python 3.10 (required by the conda environment)
- Basic understanding of machine learning concepts
- Familiarity with Python programming
- OpenAI API key (for Parts 1 and 2)
- Conda environment manager

No GPU is required - all examples can run on CPU.

## Repository Structure

```
examples/
├── README.md                                 # Overview and setup instructions
├── assignment.md                            # This assignment file
├── simple_example.py                        # Quick overview of all three approaches
├── run_all_examples.sh                      # Script to run all examples
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

## Dataset Descriptions

1. **teacher_student_dialogues.jsonl**
   - Format: Multi-turn dialogues between teacher and student
   - Size: 5-10 example conversations
   - Structure:
     ```json
     {
       "dialogue": [
         {"role": "teacher", "content": "Can you explain what photosynthesis is?"},
         {"role": "student", "content": "It's how plants make their food, right?"},
         {"role": "teacher", "content": "That's a good start! Can you tell me more about what they need to make their food?"}
       ]
     }
     ```

2. **openai_teacher_dialogue_training.jsonl**
   - Format: OpenAI chat completion format
   - Size: 15-20 training examples
   - Structure:
     ```json
     {
       "messages": [
         {"role": "system", "content": "You are a knowledgeable teacher..."},
         {"role": "user", "content": "What is the water cycle?"},
         {"role": "assistant", "content": "The water cycle is the continuous movement..."}
       ]
     }
     ```

3. **small_edu_qa.jsonl**
   - Format: Simple question-answer pairs
   - Size: 20+ examples
   - Structure:
     ```json
     {
       "instruction": "Explain the concept of gravity to a middle school student.",
       "output": "Gravity is like an invisible force that pulls objects toward each other..."
     }
     ```

## Code Structure and Implementation Details

### 1. DSPy Example (dspy_example.py)

Key components:
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
    
    def forward(self, conversation_history, teacher_prompt):
        # Generate student response using optimized prompting
        return self.chain(conversation_history=conversation_history,
                         teacher_prompt=teacher_prompt)
```

Important functions:
- `prepare_dspy_examples()`: Formats training data for DSPy
- `student_behavior_metric()`: Evaluates response quality
- `run_scenario_evaluation()`: Tests model performance

### 2. OpenAI Fine-tuning (openai_finetune.py)

Key components:
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

class FineTuningJob:
    """Manage OpenAI fine-tuning process"""
    def create_job(self):
        # Set up and monitor fine-tuning
        
    def evaluate_model(self):
        # Test fine-tuned model performance
```

### 3. HuggingFace LoRA (huggingface_lora.py)

Key components:
```python
def prepare_model_and_tokenizer():
    """Set up base model with LoRA configuration"""
    model = AutoModelForCausalLM.from_pretrained(
        "base_model",
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Apply LoRA configuration
    config = LoraConfig(
        r=8,  # Rank of update matrices
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

class EducationalDataset(Dataset):
    """Custom dataset for educational QA pairs"""
    def __init__(self, data, tokenizer):
        self.examples = self.prepare_data(data, tokenizer)
```

## Assignment Tasks

### Part 1: DSPy Prompt Optimization (40%)

1. **Understanding the Approach**
   - Run and analyze `dspy_example.py`
   - Document the chain-of-thought implementation
   - Explain how DSPy optimizes prompts without changing model weights

2. **Dataset Creation & Testing**
   - Create 5 new educational dialogues
   - Test them with the existing model
   - Document changes in performance

3. **Model Enhancement**
   - Implement one of:
     - Additional reasoning steps
     - Better evaluation metrics
     - Enhanced dialogue history handling

4. **Evaluation**
   - Compare original vs. enhanced model
   - Analyze response quality metrics
   - Document limitations and potential improvements

### Part 2: OpenAI Fine-tuning (30%)

1. **Data Preparation**
   - Create 15 domain-specific QA pairs
   - Format them for OpenAI fine-tuning
   - Validate data format

2. **Fine-tuning Setup**
   - Configure fine-tuning parameters
   - Implement monitoring and evaluation
   - (Optional) Run a test fine-tuning job

3. **Analysis**
   - Calculate costs and benefits
   - Compare with base model performance
   - Document when fine-tuning is worth it

### Part 3: HuggingFace LoRA (30%)

1. **Local Setup**
   - Configure the environment
   - Understand LoRA parameters
   - Prepare training data

2. **Implementation**
   - Modify hyperparameters for CPU training
   - Implement custom evaluation metrics
   - Test the training process

3. **Comparison**
   - Document resource usage
   - Compare with other approaches
   - Analyze tradeoffs

## Submission Requirements

1. **Code Files**
   - Modified versions of all three example files
   - Any additional utility functions created
   - Dataset files used for testing

2. **Documentation**
   - Detailed analysis of each approach
   - Performance comparisons
   - Resource usage and cost analysis

3. **Report**
   - Summary of findings
   - Recommendations for different scenarios
   - Limitations and potential improvements

## Evaluation Criteria

Your submission will be evaluated on:

1. **Technical Implementation (40%)**
   - Correct implementation of each approach
   - Code quality and documentation
   - Proper error handling

2. **Analysis & Understanding (30%)**
   - Depth of analysis
   - Understanding of tradeoffs
   - Quality of documentation

3. **Creativity & Enhancement (20%)**
   - Innovative improvements
   - Novel evaluation metrics
   - Creative problem-solving

4. **Presentation (10%)**
   - Clear documentation
   - Well-organized code
   - Professional presentation

## Resources

- DSPy Documentation: [https://dspy.ai/](https://dspy.ai/)
- OpenAI Fine-tuning Guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- HuggingFace LoRA Guide: [https://huggingface.co/docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora)

## Support

If you encounter issues:
1. Check the FAQ in README.md
2. Review the example code comments
3. Contact course staff through the designated channels

Good luck with your implementation!
