# Fine-tuning Methods for LLMs

This directory contains educational examples and assignments designed to help students learn and apply different fine-tuning approaches in their own projects. Through hands-on experience with these examples, students will develop practical skills in adapting Large Language Models (LLMs) for real-world applications.

## 📚 Learning Objectives

After working through these examples and applying them to your project, you will be able to:
1. Master three distinct fine-tuning approaches:
   - Prompt optimization with DSPy (focusing on prompt engineering and optimization)
   - Cloud-based fine-tuning with OpenAI (understanding API-based model adaptation)
   - Local parameter-efficient fine-tuning with LoRA (learning local model modification)
2. Implement each approach independently in your own project
3. Analyze and choose the best method for your specific use case based on:
   - Your project's data requirements
   - Available computational resources
   - Budget constraints
   - Performance needs
   - Implementation complexity
4. Build a production-ready fine-tuned model for your application

## 📊 Expert Evaluation Process

Your team's implementation will be evaluated through a presentation to your assigned elementary teacher professor. These experts specialize in elementary education and will assess how well your chatbot simulates a second-grade student.

### Presentation Format (20 minutes)

1. **Implementation Overview (5 minutes)**
   - Chosen fine-tuning approach(es) and rationale
   - Dataset preparation and characteristics
   - Technical implementation details
   - Challenges and solutions

2. **Live Demo (10 minutes)**
   - Interactive demonstration of your chatbot
   - Show various subject area interactions
   - Demonstrate grade-level appropriate responses
   - Handle questions from the professor

3. **Q&A Discussion (5 minutes)**
   - Address professor's questions
   - Discuss potential improvements
   - Explore educational applications
   - Receive immediate feedback

### What Professors Will Evaluate

1. **Student Simulation Authenticity**
   - Age-appropriate vocabulary and language
   - Typical second-grade reasoning patterns
   - Common misconceptions and learning challenges
   - Natural conversation flow

2. **Educational Value**
   - Alignment with second-grade curriculum
   - Learning progression demonstration
   - Subject matter understanding
   - Engagement level

### Submission Package

1. **Presentation Materials**
   - Slides covering implementation details
   - Live demo preparation
   - Example conversations ready to show
   - Technical architecture overview

2. **Implementation Documentation**
   - Source code with clear comments
   - Setup instructions
   - Training data samples
   - Configuration details

3. **Project Report**
   - Implementation approach
   - Design decisions
   - Challenges encountered
   - Future improvements

Your assigned professor will provide direct feedback during the presentation and help identify both strengths and areas for improvement in your implementation.

## 🎯 Project-Based Learning Approach

This course follows a project-based learning approach where you'll:

1. **Learn from Examples**
   - Study and run provided example implementations
   - Understand key concepts through practical demonstrations
   - Analyze real-world educational dialogue scenarios

2. **Apply to Your Project**
   - Choose a domain for your own fine-tuning project
   - Create custom datasets for your use case
   - Implement fine-tuning methods in your application
   - Evaluate and optimize performance

3. **Build Production Systems**
   - Deploy fine-tuned models
   - Monitor performance
   - Manage costs
   - Scale your solution

## 🛠️ Environment Setup

### Prerequisites
- Python 3.10 or higher
- Conda package manager
- OpenAI API key (for DSPy and OpenAI approaches)
- Git (for cloning the repository)

### Setting up the Environment

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and Activate the Conda Environment**
   ```bash
   # Create the environment from environment.yml
   conda env create -f environment.yml

   # Activate the environment
   conda activate utta
   ```

   The `environment.yml` file includes all necessary dependencies:
   - dspy-ai: For prompt optimization
   - openai: For cloud-based fine-tuning
   - torch: For local model training
   - transformers: For HuggingFace models
   - peft: For LoRA fine-tuning
   - Additional utilities and dependencies

3. **Verify Installation**
   ```bash
   # Check if key packages are installed
   python -c "import dspy; import openai; import torch; print('Setup successful!')"
   ```

### OpenAI API Key Configuration

The OpenAI API key is required for:
- DSPy approach: For making API calls during prompt optimization
- OpenAI fine-tuning: For both training and using the fine-tuned model

Set up your API key in one of these ways:

1. **Environment Variable (Recommended for Development)**
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY='your-api-key'
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY='your-api-key'
   ```

2. **Create a .env File (Recommended for Project)**
   ```bash
   # Create .env in the project root
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

3. **Direct Configuration in Python (Not Recommended)**
   ```python
   import openai
   openai.api_key = 'your-api-key'  # Not recommended for production
   ```

### Cost Management

1. **DSPy Approach**
   - API calls during optimization: ~$0.01-$0.20 per cycle
   - Can use cheaper models (e.g., GPT-3.5-turbo)
   - Best for: Quick experiments and immediate results
   - Cost reduction: Use caching and cheaper models

2. **OpenAI Fine-tuning**
   - Training: $0.008 per 1K training tokens
   - Inference: 1.5-2x base model cost
   - Best for: Production-ready educational chatbots
   - Cost reduction: Optimize dataset size, use GPT-3.5

3. **HuggingFace LoRA**
   - No API costs (runs locally)
   - Requires computational resources
   - Best for: Long-term deployment and data privacy
   - Cost reduction: Use smaller models, quantization

## 📊 Working with Datasets

### Understanding the Example Datasets

Each example is designed to demonstrate best practices that you can apply to your own project:

1. **DSPy Dataset Example**
   ```json
   {
     "dialogue": [
       {"role": "teacher", "content": "Can you explain what photosynthesis is?"},
       {"role": "student", "content": "It's how plants make their food, right?"},
       {"role": "teacher", "content": "That's a good start! Can you tell me more?"}
     ]
   }
   ```
   Apply this to your project:
   - Use multi-turn conversations for complex interactions
   - Implement progressive concept building
   - Include validation checkpoints
   - Track user engagement

2. **OpenAI Dataset Example**
   ```json
   {
     "messages": [
       {"role": "system", "content": "You are a knowledgeable teacher..."},
       {"role": "user", "content": "What is the water cycle?"},
       {"role": "assistant", "content": "The water cycle is the continuous movement..."}
     ]
   }
   ```
   Project applications:
   - Define clear system behaviors
   - Structure conversations naturally
   - Include domain-specific knowledge
   - Maintain consistent response patterns

3. **LoRA Dataset Example**
   ```json
   {
     "instruction": "Explain the concept of gravity to a middle school student.",
     "output": "Gravity is like an invisible force that pulls objects toward each other..."
   }
   ```
   Customize for your needs:
   - Adapt instruction formats
   - Target specific audience levels
   - Include domain terminology
   - Balance complexity

### Building Your Project Dataset

1. **Planning Phase**
   - Define your use case clearly
   - Identify required data types
   - Set quality standards
   - Plan data collection strategy

2. **Data Collection**
   - Gather domain-specific content
   - Record real user interactions
   - Create synthetic examples
   - Validate with experts

3. **Data Preparation**
   - Clean and normalize
   - Format for your chosen method
   - Create train/validation splits
   - Implement quality checks

4. **Iteration Process**
   - Test with small datasets
   - Gather feedback
   - Expand gradually
   - Refine based on results

## 💡 Project Implementation Guide

### 1. DSPy Implementation (Week 1-2)
1. **Setup and Exploration**
   ```python
   # Example: Custom DSPy signature for your domain
   class YourDomainSignature(dspy.Signature):
       context = dspy.InputField()
       query = dspy.InputField()
       response = dspy.OutputField()
   ```
   - Study dspy_example.py
   - Adapt signatures for your domain
   - Test basic optimizations

2. **Custom Development**
   - Create domain-specific metrics
   - Implement custom chains
   - Optimize for your use case

### 2. OpenAI Fine-tuning (Week 3-4)
1. **Initial Setup**
   ```python
   # Example: Prepare your training data
   def prepare_domain_data(your_data):
       return [{
           "messages": [
               {"role": "system", "content": YOUR_SYSTEM_PROMPT},
               {"role": "user", "content": item["input"]},
               {"role": "assistant", "content": item["output"]}
           ]
       } for item in your_data]
   ```
   - Study openai_finetune.py
   - Adapt for your domain
   - Test with small datasets

2. **Production Development**
   - Implement data pipeline
   - Set up monitoring
   - Optimize costs
   - Scale gradually

### 3. LoRA Implementation (Week 5-6)
1. **Local Setup**
   ```python
   # Example: Custom LoRA configuration
   config = LoraConfig(
       r=8,
       lora_alpha=32,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       bias="none",
       task_type="CAUSAL_LM"
   )
   ```
   - Study huggingface_lora.py
   - Configure for your resources
   - Test training pipeline

2. **Optimization**
   - Tune hyperparameters
   - Optimize memory usage
   - Implement quantization
   - Deploy efficiently

## 📝 Project Deliverables

Your project should demonstrate:

1. **Implementation (40%)**
   - Working fine-tuned model
   - Clean, documented code
   - Efficient data pipeline
   - Error handling

2. **Analysis (30%)**
   - Method comparison
   - Performance metrics
   - Cost analysis
   - Scaling considerations

3. **Innovation (20%)**
   - Novel applications
   - Creative solutions
   - Performance improvements
   - Unique features

4. **Documentation (10%)**
   - Clear explanation
   - Setup instructions
   - Usage guidelines
   - Future improvements

## 🔍 Project Examples

Here are some example projects you could develop:

1. **Customer Service Bot**
   - Fine-tune for company-specific responses
   - Handle multi-turn conversations
   - Maintain brand voice
   - Track user satisfaction

2. **Technical Documentation Assistant**
   - Process technical documents
   - Answer specific queries
   - Generate examples
   - Explain complex concepts

3. **Language Learning Tutor**
   - Adapt to learner level
   - Provide grammar corrections
   - Generate exercises
   - Track progress

4. **Healthcare Information System**
   - Process medical queries
   - Provide accurate information
   - Maintain medical guidelines
   - Handle sensitive data

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

## 📚 Resources

- DSPy Documentation: [https://dspy.ai/](https://dspy.ai/)
- OpenAI Fine-tuning Guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- HuggingFace LoRA Guide: [https://huggingface.co/docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora)

## 🤝 Support

If you encounter issues:
1. Check the documentation in each example file
2. Review the code comments
3. Contact course staff through designated channels

Good luck with your learning journey! 🚀
