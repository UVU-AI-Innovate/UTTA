# 🧠 LLM Fine-Tuning Examples for Teacher-Student Dialogues

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for realistic teacher-student dialogue interactions.
****

## 📚 Introduction

### Project Overview

This educational project demonstrates three different approaches to improving LLM performance for teacher-student dialogue interactions:

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (10-50 dialogues) | Medium (50-100+ dialogues) | Large (100-1000+ dialogues) |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU (16GB+ VRAM)* |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |
| Time to results | Immediate | Hours | Hours (with GPU) |
| Dialogue quality | Good | Better | Best |
| Evaluation metrics | Included | Included | Included |

> **Note**: For educational purposes, these examples use much smaller datasets than would be required for production use. The dataset sizes in the table reflect realistic requirements for meaningful results in real-world applications.
>
> *Students in this project can use the provided GPU server for HuggingFace fine-tuning, eliminating the hardware requirement barrier.

### Which Approach Is Right For You?

- 👉 **DSPy**: Best for quick experiments, small datasets, immediate results
- 👉 **OpenAI**: Best for production deployment, balanced control/convenience
- 👉 **HuggingFace**: Best for complete control, data privacy, long-term usage



## 🔍 Three Approaches Explained

### DSPy: Prompt Optimization

**Key Features:**
- No model weight changes - only better prompting
- Works with smaller datasets (10-50 dialogue examples)
- Uses Chain-of-Thought prompting for pedagogical reasoning
- API-based approach with immediate results
- Includes evaluation metrics for teaching quality

**How DSPy Works:**

DSPy is not traditional fine-tuning but rather a framework for optimizing prompts through a systematic approach:

1. **Modular Architecture**: DSPy defines reusable modules (like `ChainOfThought` or `Predict`) that can be composed into programs
2. **Optimization Process**:
   - Define a dialogue task as a DSPy program (e.g., teacher responses with pedagogical techniques)
   - Provide a small set of labeled teacher-student dialogue examples
   - The DSPy optimizer (Teleprompter) learns to generate better prompts by:
     - Testing different prompt variations
     - Analyzing what works and what doesn't
     - Refining prompts based on pedagogical metrics
     - Optimizing for the specific LLM being used
3. **Prompt Compilation**: The optimized prompts are "compiled" for the target LLM, improving teaching quality without changing model weights
4. **Zero-Shot Transfer**: The optimized prompts often generalize well to dialogue scenarios not seen during optimization

DSPy leverages the custom `PedagogicalMetric` to evaluate responses based on teaching techniques, follow-up questions, and explanation quality.

**How to Run:**
```bash
python dspy_example.py
```

**Cost Considerations:**
- Requires payment for each API call to the underlying LLM service
- Can use cheaper LLMs like Mixtral, Llama 2, or Claude Instant
- Typical cost: $0.01-$0.20 per optimization cycle

**Example Output:**
```
Student: I'm confused about photosynthesis.
Teacher: What specific part is confusing you? Let me help you break it down.

Student: How do plants convert sunlight to energy?
Teacher: Plants capture sunlight with chlorophyll in their chloroplasts. 
This energy is used to convert CO2 and water into glucose and oxygen 
through a series of chemical reactions. Does that help clarify the 
process, or would you like me to explain a specific part in more detail?
```

### OpenAI: Cloud Fine-Tuning

**Key Features:**
- Updates actual model weights through cloud service
- Requires medium-sized datasets (50-100+ dialogue examples)
- Balance of control and convenience
- Handles multi-turn conversations with context preservation
- Comprehensive evaluation framework for teaching quality

**How to Run:**
```bash
python openai_finetune.py
```

**Cost Considerations:**
- One-time training fee ($0.008 per 1K training tokens)
- Ongoing API costs for inference (1.5-2x base model cost)
- GPT-3.5 Turbo is significantly cheaper than GPT-4
- Typical cost: $2-$5 for training small dialogue datasets + ongoing inference costs

**Example Output:**
```
Student: I don't understand Newton's third law.
Teacher: Newton's third law states that for every action, there's an equal 
and opposite reaction. Can you think of any examples where you might have 
experienced this in everyday life?

Student: Maybe when I push against a wall?
Teacher: Exactly! When you push against a wall, you're applying a force to 
the wall, but at the same time, the wall is pushing back against your hand 
with equal force. That's why your hand doesn't go through the wall. Can you 
think of another example?
```

### HuggingFace: Local Fine-Tuning with LoRA

**Key Features:**
- Full local control over model and training process
- Uses LoRA for efficient adaptation of model weights
- Works best with larger datasets (100-1000+ dialogue examples)
- Data privacy - all information stays on your machine
- **Access**: Students can use the provided GPU server for fine-tuning

**How to Run:**
```bash
python huggingface_lora.py
```

**Cost Considerations:**
- One-time compute cost for GPU usage, no ongoing API fees
- Can use smaller open-source models and lower precision (int8/int4)
- For this project: No cloud costs needed as students have access to the GPU server

**Example Output:**
```
Student: I'm struggling with calculating derivatives.
Teacher: I understand derivatives can be challenging. Could you tell me 
which specific aspect you're finding difficult?

Student: I don't understand the chain rule.
Teacher: The chain rule helps us find the derivative of composite functions. 
If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Let's work through an example: 
if y = sin(x²), what would be its derivative?
```

---

## 💡 Technical Deep Dive: Fine-Tuning Evaluation

Each example now includes comprehensive evaluation metrics to assess the quality of teacher-student interactions:

### Evaluation Metrics Explained

1. **Pedagogical Techniques**: Measures the percentage of responses that include effective teaching techniques like Socratic questioning, scaffolding, and guided inquiry
2. **Follow-up Questions**: Tracks how often the model asks questions to check student understanding or guide their thinking
3. **Response Length**: Evaluates the level of detail and thoroughness in explanations
4. **Adaptability**: Assesses the model's ability to adjust explanations based on student's previous statements

### Evaluation Implementation

Both DSPy and OpenAI examples include built-in evaluation that:
- Tests on standardized scenarios across different subjects
- Provides side-by-side comparisons of performance before and after fine-tuning
- Calculates quantitative improvements across all metrics
- Includes interactive demonstration mode for testing custom queries

Example evaluation output:
```
--- Evaluation Results Comparison ---
Metric                    Unoptimized      Optimized        Improvement     
----------------------------------------------------------------------
pedagogical_techniques    45.00            75.00            +30.00         
follow_up_questions       50.00            83.33            +33.33         
response_length           215.67           268.50           +52.83         
adaptability              16.67            33.33            +16.66
```

---

## 💰 Cost Analysis

### Side-by-Side Comparison

| Cost Factor | DSPy | OpenAI | HuggingFace |
|-------------|------|--------|-------------|
| Initial cost | None | $2-$5 (training) | GPU hardware/rental |
| Ongoing costs | API calls for each dialogue | API calls for each dialogue | None (self-hosted) |
| Cost reduction options | Use cheaper models (Mixtral, Llama) | Use GPT-3.5 instead of GPT-4 | Use smaller models/lower precision |
| Data usage costs | API calls during optimization | Training + API inference | None |
| Pricing model | Pay-per-request | Training + pay-per-request | One-time compute |
| Best for budget | Small, quick experiments | Medium-term projects | Long-term, high-volume usage |

### Budget Considerations

- **With GPU Server Access**: HuggingFace approach is cheapest overall (students in this project have access to the GPU server)
- **Without GPU Access**: DSPy with budget-friendly models is most affordable
- **For Production**: Calculate based on expected conversation volume and frequency

---

## 🤖 Alternative LLM Options

When working with these approaches, you can use different LLMs to optimize for cost, performance, and other factors:

### Mixtral 8x7B
- **What it is**: An open-weights mixture-of-experts model developed by Mistral AI
- **Architecture**: Uses a sparse mixture-of-experts approach where only a subset of parameters activate for each token
- **Performance**: Competitive with much larger models while being more efficient
- **Cost**: Can be used via Mistral API (~$0.0002/1K tokens) or self-hosted
- **Best for**: DSPy or HuggingFace approaches where you need strong reasoning at lower cost
- **Availability**: Commercial use allowed

### Llama 2
- **What it is**: Meta's open-source model family available in multiple sizes (7B, 13B, 70B parameters)
- **Architecture**: Traditional transformer architecture with improvements from the original Llama
- **Performance**: The 70B variant approaches GPT-3.5 performance for many tasks
- **Cost**: Free for self-hosting; also available through various APIs
- **Best for**: HuggingFace approach where you can fine-tune locally
- **Sizes**: 
  - 7B: Fastest, lowest memory requirements (16GB VRAM)
  - 13B: Better quality, moderate requirements (24GB VRAM)
  - 70B: Best quality, high requirements (80GB+ VRAM or quantized)
- **Availability**: Commercial use allowed with a license

### Claude Instant
- **What it is**: Anthropic's faster, more cost-effective version of their Claude model
- **Architecture**: Proprietary architecture with constitutional AI alignment
- **Performance**: Good balance of reasoning capabilities and speed
- **Cost**: ~$0.008/1K tokens (input), ~$0.024/1K tokens (output)
- **Best for**: DSPy approach when you need strong reasoning but GPT-4 is too expensive
- **Availability**: Only available through Anthropic's API

### LLM Comparison

| Factor | Mixtral 8x7B | Llama 2 | Claude Instant |
|--------|-------------|---------|----------------|
| API Cost | Low | Very low/Free | Medium |
| Self-hostable | Yes | Yes | No |
| Reasoning ability | High | Medium-High | High |
| Context length | 32K tokens | 4K tokens | 100K tokens |
| Best paired with | DSPy/HuggingFace | HuggingFace | DSPy |

---

## 🛠️ Implementation Details

### Getting Started

All dependencies for these examples are included in the main project's `environment.yml` file.

```bash
# Create and activate conda environment from the project root
conda env create -f environment.yml
conda activate utta
```

### Example Files
- **DSPy Optimization**: [dspy_example.py](dspy_example.py)
- **OpenAI Fine-Tuning**: [openai_finetune.py](openai_finetune.py)
- **HuggingFace LoRA**: [huggingface_lora.py](huggingface_lora.py)

### Datasets
All examples use consistent dialogue datasets for fair comparison:
- **Base dataset**: 5 teacher-student dialogue sequences (used in all examples)
- **Extended dataset**: 10-20 dialogue sequences (used in OpenAI and HuggingFace examples)
- Data formats:
  - DSPy uses [teacher_student_dialogues.jsonl](teacher_student_dialogues.jsonl) (created on first run)
  - OpenAI uses [openai_teacher_dialogue_training.jsonl](openai_teacher_dialogue_training.jsonl)

### Dataset Formats & Creation

Each approach requires datasets in specific formats. Here's how to prepare data for each method:

#### DSPy Dataset Format

DSPy dialogue datasets are stored in JSONL format, with each line containing a dialogue sequence:

```json
{
  "dialogue": [
    {"role": "student", "content": "I'm confused about photosynthesis."},
    {"role": "teacher", "content": "What specific part is confusing you?"},
    {"role": "student", "content": "How do plants convert sunlight to energy?"},
    {"role": "teacher", "content": "Plants capture sunlight with chlorophyll in their chloroplasts. This energy is used to convert CO2 and water into glucose and oxygen through a series of chemical reactions."}
  ]
}
```

**Creating a DSPy Dataset**:
1. Collect sequences of student-teacher interactions
2. Format each dialogue as shown above with alternating role/content pairs
3. Combine multiple examples into a JSONL file (one dialogue per line)
4. For best results, include diverse teaching scenarios and techniques

DSPy works best with exemplary teaching dialogues that demonstrate effective pedagogical strategies.

#### OpenAI Fine-Tuning Dataset Format

OpenAI requires dialogues in the chat completions format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful teacher assisting a student."},
    {"role": "user", "content": "I'm struggling with calculating derivatives."},
    {"role": "assistant", "content": "I understand derivatives can be challenging. Could you tell me which specific aspect you're finding difficult?"},
    {"role": "user", "content": "I don't understand the chain rule."},
    {"role": "assistant", "content": "The chain rule helps us find the derivative of composite functions. If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Let's work through an example: if y = sin(x²), what would be its derivative?"}
  ]
}
```

**Creating an OpenAI Dataset**:
1. Format your data with the system, user, and assistant roles
2. Include a system message that defines the teaching approach
3. Structure the conversation as alternating user (student) and assistant (teacher) messages
4. Combine examples into a JSONL file
5. Validate using OpenAI's CLI tool: `openai tools fine_tunes.prepare_data -f your_data.jsonl`

For effective fine-tuning, include examples of the teaching strategies you want the model to learn.

#### HuggingFace/LoRA Dataset Format

HuggingFace typically uses datasets with clear input/output pairs:

```json
{
  "input": "Student: I don't understand Newton's third law.\nTeacher:", 
  "output": "Newton's third law states that for every action, there's an equal and opposite reaction. When you push against a wall, the wall pushes back with equal force. Can you think of examples where you've experienced this in everyday life?"
}
```

**Creating a HuggingFace Dataset**:
1. Format your data with consistent input/output fields
2. For instruction-tuning, include clear context in the input
3. Save as a JSON/JSONL or CSV file
4. Consider using the Hugging Face Datasets library for processing and formatting
5. For better results, include diverse teaching approaches in the training data

In production, you would typically need hundreds to thousands of examples for effective LoRA fine-tuning.

### Building Realistic Teacher-Student Interaction Datasets

For educational chatbots that simulate realistic teacher-student interactions, simple Q&A pairs are often insufficient. Here's how to create more authentic conversational datasets for each approach:

#### Multi-turn Conversations

Instead of single question-answer pairs, structure your data as complete multi-turn conversations:

**DSPy Realistic Format Example:**
```json
{"conversation": [
  {"role": "student", "content": "I'm confused about photosynthesis."},
  {"role": "teacher", "content": "What specific part is confusing you?"},
  {"role": "student", "content": "How do plants convert sunlight to energy?"},
  {"role": "teacher", "content": "Plants capture sunlight with chlorophyll in their chloroplasts. This energy is used to convert CO2 and water into glucose and oxygen through a series of chemical reactions."}
]}
```

**OpenAI Realistic Format Example:**
```json
{"messages": [
  {"role": "system", "content": "You are a helpful teacher assisting a student."},
  {"role": "user", "content": "I'm struggling with calculating derivatives."},
  {"role": "assistant", "content": "I understand derivatives can be challenging. Could you tell me which specific aspect you're finding difficult?"},
  {"role": "user", "content": "I don't understand the chain rule."},
  {"role": "assistant", "content": "The chain rule helps us find the derivative of composite functions. If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Let's work through an example: if y = sin(x²), what would be its derivative?"}
]}
```

**HuggingFace Realistic Format Example:**
```json
{"input": "Student: I don't understand Newton's third law.\nTeacher:", 
 "output": "Newton's third law states that for every action, there's an equal and opposite reaction. When you push against a wall, the wall pushes back with equal force. Can you think of examples where you've experienced this in everyday life?"}
```

#### Pedagogical Elements to Include

For realistic teacher-student interactions, your dataset should include examples of:

1. **Scaffolding techniques:** Gradually building understanding through structured support
2. **Socratic questioning:** Using questions to guide students toward discovering answers
3. **Clarification requests:** Asking students to elaborate on their understanding
4. **Misconception handling:** Identifying and correcting common misconceptions
5. **Knowledge checks:** Verifying student understanding during the conversation
6. **Worked examples:** Walking through problems step-by-step
7. **Real-world applications:** Connecting abstract concepts to concrete examples
8. **Metacognitive prompts:** Encouraging students to reflect on their learning process

#### Data Collection Strategies

To build effective teacher-student interaction datasets:

1. **Observe real interactions:** Record (with permission) actual teacher-student conversations
2. **Role-play scenarios:** Have educators simulate typical teaching interactions
3. **Analyze textbooks:** Extract explanation patterns from educational materials
4. **Include diverse subjects:** Cover various subjects and difficulty levels
5. **Represent different learning styles:** Visual, auditory, and kinesthetic approaches
6. **Balance guidance levels:** Include both heavily guided and more independent learning examples

By incorporating these elements, your chatbot will better simulate the nuanced, adaptive nature of effective teaching rather than simply providing answers to questions.

### Implementation Structure

All examples follow the same educational structure:

1. **Environment & Data Preparation**: Setup and dialogue dataset preparation
2. **Data Format Preparation**: Converting data to the required format
3. **Model Configuration**: Setting up the model architecture with pedagogical focus
4. **Fine-Tuning Implementation**: The optimization or training process
5. **Comprehensive Evaluation**: Measuring teaching quality before and after fine-tuning
6. **Interactive Demonstration**: Testing the dialogue capabilities
7. **Comparison**: Direct comparison with other methods

All examples include API key verification, proper error handling, and safety measures to prevent accidental execution that might incur costs.

---

## 🧭 Decision Guide

Use this guide to quickly identify which approach best fits your specific needs:

### By Dataset Size

- **Small (10-50 dialogues)** → DSPy
- **Medium (50-100+ dialogues)** → OpenAI
- **Large (100-1000+ dialogues)** → HuggingFace

### By Time Constraints

- **Need immediate results** → DSPy
- **Can wait hours** → OpenAI/HuggingFace

### By Budget

- **Minimal budget** → DSPy (still requires paying for API calls)
- **Medium budget** → OpenAI
- **One-time compute cost** → HuggingFace
- **Cheapest overall**: HuggingFace with existing GPU (recommended for this project)
- **Cheapest without GPU**: DSPy with budget-friendly models

### By Privacy Requirements

- **Standard** → DSPy/OpenAI
- **High privacy needs** → HuggingFace
