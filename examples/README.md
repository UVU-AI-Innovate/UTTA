# 🎓 Fine-tuning LLMs for Educational Applications

This guide provides a structured approach to understanding, implementing, and evaluating different fine-tuning methods for Large Language Models (LLMs), with a focus on simulating a second-grade student chatbot.

## 📋 Table of Contents

1. [Learning Objectives](#-learning-objectives)
2. [Fine-tuning Methods Comparison](#-fine-tuning-methods-comparison)
3. [Dataset Requirements](#-dataset-requirements)
4. [Implementation Process](#-implementation-process)
5. [Environment Setup](#-environment-setup)
6. [Dataset Creation Guide](#-dataset-creation-guide)
7. [Detailed Implementation](#-detailed-implementation)
8. [Evaluation Process](#-evaluation-process)
9. [Resources & Support](#-resources--support)

## 🎯 Learning Objectives

After working with these examples, you will understand:

| Knowledge Area | Skills Developed |
|----------------|------------------|
| **Technical** | • Compare three distinct fine-tuning approaches<br>• Analyze tradeoffs in data requirements<br>• Implement appropriate method for your use case |
| **Educational** | • Simulate age-appropriate responses<br>• Demonstrate grade-level reasoning<br>• Model learning progression |
| **Practical** | • Create effective training datasets<br>• Optimize model performance<br>• Deploy and evaluate your solution |

## 🔄 Fine-tuning Methods Comparison

Each method represents a different approach to adapting an LLM for a specific purpose:

| Method | DSPy | OpenAI Fine-tuning | LoRA |
|--------|------|-------------------|------|
| **What It Does** | Optimizes prompts without changing model weights | Updates model weights through API | Efficiently adapts specific model parameters locally |
| **Data Needed** | 10-50 examples | 50-100+ examples | 100-1000+ examples |
| **Technical Complexity** | Low-Medium | Low | Medium-High |
| **Infrastructure** | API access only | API access only | Local GPU required |
| **Cost Model** | Pay per API call | Training fee + higher inference | One-time compute cost |
| **Permanence** | Temporary (prompt-based) | Permanent changes | Permanent but modular |
| **Best For** | Quick prototyping<br>Limited examples<br>Testing concepts | Production chatbots<br>Consistent behavior<br>Cloud deployment | Complete control<br>Data privacy<br>Customization |

## 📊 Dataset Requirements

The structure of your dataset is critical to successful fine-tuning. Each method requires a specific format:

### Dataset Format Requirements

| Method | Format | Structure | Example |
|--------|--------|-----------|---------|
| **DSPy** | Dialogue format | Multi-turn dialogues with teacher-student pairs | [View Format](#dspy-format) |
| **OpenAI** | Message format | Role-based messages with system prompt | [View Format](#openai-format) |
| **LoRA** | QA format | Simple question-answer pairs | [View Format](#lora-format) |

<a id="dspy-format"></a>
### DSPy Format (teacher_student_dialogues.jsonl)

```json
{
  "dialogue": [
    {"role": "teacher", "content": "What happens when ice melts?"},
    {"role": "student", "content": "It turns into water! We did an experiment in class where we put ice cubes in the sun."},
    {"role": "teacher", "content": "Great observation! Why do you think the ice melted in the sun?"},
    {"role": "student", "content": "Because the sun is hot! My teacher said heat makes ice melt."}
  ]
}
```

<a id="openai-format"></a>
### OpenAI Format (openai_teacher_dialogue_training.jsonl)

```json
{
  "messages": [
    {"role": "system", "content": "You are a second-grade student with age-appropriate vocabulary and understanding."},
    {"role": "user", "content": "What is the water cycle?"},
    {"role": "assistant", "content": "Oh, we learned about that! Water goes up to the sky and makes clouds. My teacher calls it eva... evapuration when the sun makes water go up."}
  ]
}
```

<a id="lora-format"></a>
### LoRA Format (small_edu_qa.jsonl)

```json
{
  "question": "How do plants grow?",
  "answer": "Plants grow from seeds! First you put a seed in dirt and water it. Then it gets little roots under the ground. After that, it grows up through the dirt and gets leaves."
}
```

## 🔄 Implementation Process

The implementation process follows these key steps for each fine-tuning method:

| Stage | DSPy | OpenAI Fine-tuning | LoRA |
|-------|------|-------------------|------|
| **Preparation** | Create dialogue dataset | Format messages with system role | Prepare QA pairs |
| **Development** | Define signatures & modules | Upload dataset via API | Configure model parameters |
| **Training** | Run prompt optimization | Initiate fine-tuning job | Execute training script |
| **Deployment** | Use optimized prompts | Deploy fine-tuned model | Load adapted model |
| **Evaluation** | Test with novel questions | Validate response consistency | Measure performance metrics |
| **Timeline** | 1-2 days | 2-5 days | 3-7 days |

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
   
   **Key Dependencies**:
   | Package | Purpose | Used For |
   |---------|---------|----------|
   | dspy-ai | Prompt optimization | DSPy approach |
   | openai | API access | OpenAI & DSPy approaches |
   | torch | Machine learning framework | LoRA approach |
   | transformers | Model implementations | LoRA approach |
   | peft | Parameter-efficient fine-tuning | LoRA approach |

3. **OpenAI API Key Configuration**

   The OpenAI API key is required for:
   - DSPy approach: For making API calls during prompt optimization
   - OpenAI fine-tuning: For both training and using the fine-tuned model

   **Configuration Options**:
   ```bash
   # Linux/macOS
   export OPENAI_API_KEY='your-api-key'
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY='your-api-key'
   
   # Using a .env file
   echo "OPENAI_API_KEY=your-api-key" > .env
   ```

### Cost Considerations

| Method | Cost Type | Typical Expense | Cost Reduction Strategies |
|--------|-----------|----------------|---------------------------|
| **DSPy** | API calls | $0.01-$0.20 per cycle | Use caching, cheaper models |
| **OpenAI** | Training + Inference | $0.008/1K tokens + 1.5-2x base inference | Optimize dataset size, use GPT-3.5 |
| **LoRA** | Compute resources | One-time compute cost | Use smaller models, quantization |

## 📊 Dataset Creation Guide

### Technical Requirements by Method

| Aspect | DSPy | OpenAI | LoRA |
|--------|------|--------|------|
| **Format** | Multi-turn dialogues | Role-based messages | QA pairs |
| **Size** | 10-50 examples | 50-100+ examples | 100-1000+ examples |
| **Structure** | Teacher-student turns | System-user-assistant | Question-answer |
| **Processing** | Extracts turn pairs | Used directly | Transformed to instructions |
| **Critical Element** | Context flow | System message | Answer style |

### Creating Datasets for Second-Grade Student Simulation

1. **Educational Content Requirements**
   - Use vocabulary level of 2,000-5,000 words
   - Include age-appropriate reasoning (concrete, not abstract)
   - Show partial understanding of complex concepts
   - Demonstrate common misconceptions
   - Use simple sentence structures with occasional errors

2. **Subject Coverage**
   
   | Subject | Topics | Example Questions |
   |---------|--------|-------------------|
   | **Math** | Basic addition/subtraction<br>Simple fractions<br>Time and money | "What is 15+8?"<br>"How do you share 3 cookies with 2 friends?"<br>"How many quarters make a dollar?" |
   | **Science** | Plants and animals<br>Weather<br>States of matter | "How do plants grow?"<br>"Why does it rain?"<br>"What happens to water when it freezes?" |
   | **Language Arts** | Reading comprehension<br>Basic grammar<br>Storytelling | "What is your favorite book?"<br>"What are nouns and verbs?"<br>"Can you tell me a short story?" |
   | **Social Studies** | Communities<br>Jobs<br>Geography | "What jobs help our community?"<br>"What are the seasons?"<br>"What are some different places people live?" |

3. **Technical Validation Checklist**
   - ✓ Follows the exact format required by your chosen method
   - ✓ Contains age-appropriate content and language
   - ✓ Includes a variety of subjects and question types
   - ✓ Each example is valid JSON without formatting errors
   - ✓ Files use JSONL format (one JSON object per line)

## 💡 Detailed Implementation

### 1. DSPy Implementation

**Why This Structure Works:**
- **Pattern Learning**: DSPy identifies patterns in dialogue to create optimized prompts
- **Context Utilization**: Multi-turn format shows how context influences responses
- **Extraction Process**: Code extracts teacher-student pairs to learn prompt-response relationships

**Code Implementation:**
```python
def prepare_dspy_examples(dialogues):
    examples = []
    for dialogue_data in dialogues:
        dialogue = dialogue_data["dialogue"]
        for i in range(1, len(dialogue), 2):
            if i >= len(dialogue) - 1:
                break
            teacher_prompt = dialogue[i-1]["content"]
            student_response = dialogue[i]["content"]
            # Get conversation history up to this point
            history = format_dialogue_history(dialogue, i-2) if i > 1 else ""
            example = dspy.Example(
                conversation_history=history,
                teacher_prompt=teacher_prompt,
                student_response=student_response
            )
            examples.append(example)
    return examples
```

**Implementation Steps:**
1. Create your dialogue dataset showing second-grade student responses
2. Define your DSPy signature and module structure
3. Set up optimization metrics (age appropriateness, reasoning patterns)
4. Run the optimization process
5. Use the optimized prompt templates in your application

### 2. OpenAI Fine-tuning

**Why This Structure Is Required:**
- **API Requirement**: OpenAI's fine-tuning API strictly requires the message format
- **Role-Based Learning**: The role designations are critical for model understanding
- **System Message**: Defines the global behavior and personality for all responses

**Code Implementation:**
```python
def prepare_training_data(your_data):
    return [{
        "messages": [
            {"role": "system", "content": "You are a second-grade student who..."},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]}
        ]
    } for item in your_data]
```

**Implementation Steps:**
1. Create your dataset with proper message format
2. Ensure the system message clearly defines second-grade behavior
3. Upload your dataset using the OpenAI API
4. Initiate the fine-tuning job
5. Deploy and test the fine-tuned model

### 3. LoRA Fine-tuning

**Why This Structure Works:**
- **Parameter Efficiency**: LoRA requires simple pairs for focused adaptation
- **Training Efficiency**: Simpler format enables more efficient processing
- **Internal Transformation**: Code converts QA pairs to instruction format

**Code Implementation:**
```python
def format_prompt(example):
    """Format prompt with instruction template"""
    return f"<s>[INST] You are a second-grade student. Answer this question: {example['question']} [/INST]\n\n{example['answer']}</s>"

# LoRA Configuration
config = LoraConfig(
    r=8,  # Rank of update matrices
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Implementation Steps:**
1. Prepare your question-answer dataset
2. Configure LoRA parameters based on your GPU capabilities
3. Train the model locally
4. Save the adapter weights
5. Load and test the adapted model

## 📊 Evaluation Process

Your team's implementation will be evaluated through a presentation to your assigned elementary teacher professor.

### Presentation Format (20 minutes)

| Section | Duration | Content |
|---------|----------|---------|
| **Implementation Overview** | 5 minutes | • Chosen fine-tuning approach(es)<br>• Dataset preparation<br>• Technical implementation<br>• Challenges and solutions |
| **Live Demo** | 10 minutes | • Interactive demonstration<br>• Various subject interactions<br>• Grade-level responses<br>• Professor questions |
| **Q&A Discussion** | 5 minutes | • Address professor questions<br>• Discuss improvements<br>• Explore applications<br>• Receive feedback |

### Evaluation Criteria

| Aspect | What Professors Will Evaluate |
|--------|------------------------------|
| **Student Simulation Authenticity** | • Age-appropriate vocabulary<br>• Typical reasoning patterns<br>• Common misconceptions<br>• Natural conversation flow |
| **Educational Value** | • Curriculum alignment<br>• Learning progression<br>• Subject understanding<br>• Engagement level |

## 📚 Resources & Support

| Resource | URL | Purpose |
|----------|-----|---------|
| DSPy Documentation | [https://dspy.ai/](https://dspy.ai/) | DSPy implementation details |
| OpenAI Fine-tuning | [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) | OpenAI fine-tuning guide |
| HuggingFace LoRA | [https://huggingface.co/docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora) | LoRA implementation |

### Repository Structure

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

If you encounter issues:
1. Check the documentation in each example file
2. Review the code comments
3. Contact course staff through designated channels

Good luck with your learning journey! 🚀
