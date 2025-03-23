# 🧠 LLM Fine-Tuning Examples for Teacher-Student Dialogues

This directory contains examples of three different approaches to fine-tuning large language models (LLMs) for realistic teacher-student dialogue interactions.

## 📚 Introduction

### Three Fine-Tuning Approaches

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (10-50 dialogues) | Medium (50-100+ dialogues) | Large (100-1000+ dialogues) |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU (16GB+ VRAM) |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |
| Time to results | Immediate | Hours | Hours (with GPU) |

## 🔍 Example Files

### Core Example Files
- `dspy_example.py` - DSPy prompt optimization for teacher-student dialogue
- `openai_finetune.py` - OpenAI fine-tuning for educational dialogues
- `huggingface_lora.py` - HuggingFace LoRA fine-tuning with local models

### Data Files
- `teacher_student_dialogues.jsonl` - Sample dialogue data for DSPy
- `small_edu_qa.jsonl` - Sample Q&A pairs for simple examples
- `openai_edu_qa_training.jsonl` - Sample training data for OpenAI fine-tuning

### Utility Scripts
- `run_all_examples.sh` - Script to run all three main examples
- `simple_example.py` - Quick overview of the three fine-tuning approaches

### Assignment
- `assingment.md` - Detailed assignment instructions for students

## 🚀 Quick Start

To understand the basic concepts of each approach:

```bash
python simple_example.py
```

To run the detailed examples (note: requires API keys and/or GPU):

```bash
./run_all_examples.sh
```

## 🔍 Approach Details

### DSPy: Prompt Optimization

DSPy optimizes the prompts used with LLMs without changing the model weights. This approach:
- Requires minimal training data
- Gives immediate results
- Works with any LLM through APIs
- Uses Chain-of-Thought prompting for improved pedagogical responses

### OpenAI: Cloud Fine-Tuning

OpenAI fine-tuning updates model weights through their cloud service:
- Requires more training data than DSPy
- Handles multi-turn conversations well
- Provides better specialized responses than prompt engineering alone
- Needs no local GPU infrastructure

### HuggingFace: Local LoRA Fine-Tuning

HuggingFace's LoRA (Low-Rank Adaptation) approach:
- Updates model weights locally
- Gives complete control over the training process
- Keeps all data private on your own hardware
- Requires GPU for efficient training
- Produces models that can be deployed anywhere

## 🔧 Which Approach Is Right For You?

- **DSPy**: Best for quick experiments, small datasets, immediate results
- **OpenAI**: Best for production deployment, balanced control/convenience
- **HuggingFace**: Best for complete control, data privacy, long-term usage

---

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

## 📊 Input, Output, and Evaluation Framework

### Input Datasets Overview

Each fine-tuning approach uses teacher-student dialogue data but with specific format requirements:

| Aspect | DSPy | OpenAI | HuggingFace |
|--------|------|--------|-------------|
| **Format** | JSONL with dialogue arrays | JSONL with messages arrays | JSONL/CSV with input/output pairs |
| **Minimum examples** | 5-10 high-quality dialogues | 15-20 diverse dialogues | 30+ diverse dialogues |
| **Conversation turns** | Multi-turn (2-4 exchanges) | Multi-turn (2-6 exchanges) | Usually single-turn or dual-turn |
| **Essential fields** | role, content | role (system, user, assistant), content | input, output |
| **System message** | Not required | Required for context | Embedded in input |
| **Special features** | Focus on pedagogical techniques | Include teaching style variations | Include detailed explanations |

### Input → Processing → Output Flow

Here's how the data flows through each system:

```
┌─── DSPy ───────────────────────────┐
│                                    │
│  [Teacher-Student Dialogues]       │
│          │                         │
│          ▼                         │
│  [DSPy Optimization Process]       │
│          │                         │
│          ▼                         │
│  [Improved Prompting]              │
│          │                         │
│          ▼                         │
│  [Enhanced Teacher Responses]      │
│                                    │
└────────────────────────────────────┘

┌─── OpenAI ─────────────────────────┐
│                                    │
│  [Teacher-Student Dialogues]       │
│          │                         │
│          ▼                         │
│  [Cloud Fine-Tuning]               │
│          │                         │
│          ▼                         │
│  [Updated Model Weights]           │
│          │                         │
│          ▼                         │
│  [Customized Teacher Responses]    │
│                                    │
└────────────────────────────────────┘

┌─── HuggingFace ──────────────────────┐
│                                      │
│  [Teacher-Student Dialogues]         │
│          │                           │
│          ▼                           │
│  [LoRA Parameter-Efficient Training] │
│          │                           │
│          ▼                           │
│  [Adapter Weights]                   │
│          │                           │
│          ▼                           │
│  [Self-Hosted Teacher AI]            │
│                                      │
└──────────────────────────────────────┘
```

### Expected Outputs

Each approach aims to produce teacher responses with specific qualities:

| Quality | Description | Example |
|---------|-------------|---------|
| **Pedagogical techniques** | Uses educational strategies like Socratic questioning | "What do you think might happen if we increase the temperature?" |
| **Scaffolding** | Building understanding in steps | "Let's start with the basics before tackling the complex concept." |
| **Knowledge checks** | Verifying student understanding | "Does that explanation make sense to you? Can you summarize what we just discussed?" |
| **Misconception handling** | Tactfully addressing incorrect ideas | "That's a common misunderstanding. Let me clarify why..." |
| **Personalization** | Adapting to the student's level | "Since you're familiar with X, we can connect this new concept to it." |

### Comprehensive Evaluation Metrics

Our examples include standardized evaluation across 4 key dimensions:

#### 1. Pedagogical Techniques Score (0-100%)
Measures the percentage of responses employing evidence-based teaching strategies:
- Socratic questioning
- Scaffolding concepts
- Guided discovery
- Analogies and metaphors
- Examples and counterexamples

**Calculation:** `(responses using techniques / total responses) × 100%`

#### 2. Follow-up Questions Rate (0-100%)
Tracks how often the model asks questions to check understanding:
- Comprehension checks
- Application questions
- Analysis prompts
- Reflection questions

**Calculation:** `(responses containing questions / total responses) × 100%`

#### 3. Response Quality Metrics
Quantitative measures of response characteristics:
- **Length:** Average character count (optimal range: 200-500 characters)
- **Complexity:** Flesch-Kincaid readability score (optimal range varies by grade level)
- **Specificity:** Use of subject-specific terminology (higher is better)

#### 4. Adaptability Score (0-100%)
Assesses the model's ability to adjust to the student's prior statements:
- References to student's previous comments
- Adjustment of explanation level
- Responsiveness to confusion signals

**Calculation:** `(responses showing adaptation / total responses) × 100%`

### Sample Evaluation Output

```
--- Detailed Evaluation Results ---
Subject: Physics
Total dialogues evaluated: 5
Total turns evaluated: 15

Pedagogical Metrics:
┌────────────────────────┬─────────────┬────────────┬─────────────┐
│ Metric                 │ Before      │ After      │ Change      │
├────────────────────────┼─────────────┼────────────┼─────────────┤
│ Pedagogical techniques │ 45.0%       │ 75.0%      │ +30.0%      │
│ Follow-up questions    │ 50.0%       │ 83.3%      │ +33.3%      │
│ Response length        │ 215.7 chars │ 268.5 chars│ +52.8 chars │
│ Adaptability           │ 16.7%       │ 33.3%      │ +16.6%      │
└────────────────────────┴─────────────┴────────────┴─────────────┘

Performance by Subject:
┌────────────┬─────────────────┬────────────────┐
│ Subject    │ Pre-Optimization│ Post-Optimizat.│
├────────────┼─────────────────┼────────────────┤
│ Biology    │ 42.5%           │ 72.5%          │
│ Physics    │ 44.0%           │ 78.0%          │
│ Mathematics│ 48.5%           │ 77.5%          │
└────────────┴─────────────────┴────────────────┘
```

### Interpreting Evaluation Results

- **Strong Results**: Overall scores >70% indicate effective teaching interactions
- **Areas for Improvement**: 
  - Low follow-up question rates (<50%) suggest insufficient engagement
  - Short responses (<150 chars) may lack necessary detail
  - Low adaptability (<30%) indicates generic rather than personalized responses
- **Subject Variations**: Performance differences across subjects help identify knowledge gaps

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
