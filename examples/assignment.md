# Introduction

In this assignment, you will explore three different approaches to improving LLM performance on a specific task: educational question answering. You'll work with the provided examples to gain hands-on experience with prompt optimization, cloud-based fine-tuning, and local fine-tuning with parameter-efficient techniques.

All files needed for this assignment are available at: [https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples](https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples)

## Learning Objectives

- Understand the key differences between prompt optimization, cloud fine-tuning, and local fine-tuning
- Gain practical experience with DSPy, OpenAI fine-tuning, and HuggingFace LoRA
- Analyze tradeoffs in data requirements, computational resources, and model performance
- Develop skills in evaluating which approach is most suitable for different scenarios

## Prerequisites

- Python 3.10 (required by the conda environment)
- Access to the example files: `dspy_example.py`, `openai_finetune.py`, `huggingface_lora.py`, and `simple_example.py`
- OpenAI API key (for Parts 1 and 2)
- Conda environment manager

**Important Note**: No GPU is required for any part of this assignment. All examples and tasks can be completed on any standard computer using the provided conda environment.

### Environment Setup

We've provided an `environment.yml` file with all necessary dependencies. To set up your environment:

1. **Install Conda** if you don't have it already: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)

2. **Create and activate the conda environment**:
   ```bash
   # From the project root directory
   conda env create -f environment.yml
   conda activate utta
   ```

3. **Verify installation**:
   ```bash
   # Check that key packages are available
   python -c "import dspy; import torch; import openai; print('Environment successfully configured!')"
   ```

The environment includes:
- Python 3.10
- PyTorch 2.2.0
- DSPy 2.0.4
- OpenAI 0.28.0
- Other necessary libraries for the examples

### OpenAI API Setup

For Parts 1 and 2, you'll need to set up your OpenAI API key:

1. **Get an API key** from [OpenAI's platform](https://platform.openai.com/api-keys) if you don't already have one

2. **Create a .env file** in the project root directory:
   ```
   # .env file contents
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Test your API key**:
   ```bash
   python -c "import os; from dotenv import load_dotenv; import openai; load_dotenv(); openai.api_key = os.getenv('OPENAI_API_KEY'); print('API key loaded successfully!')"
   ```

4. **Cost Management**: To minimize costs, use the `gpt-3.5-turbo` model for all API calls. This model is specified in the examples and provides a good balance of capabilities and cost-effectiveness.

## Assignment Overview

This assignment is structured in three parts of increasing complexity:

1. **Part 1: DSPy Prompt Optimization** - Modify and extend the DSPy example
2. **Part 2: OpenAI Fine-Tuning** - Prepare data and set up an OpenAI fine-tuning job
3. **Part 3: HuggingFace LoRA Fine-Tuning** - Run a parameter-efficient fine-tuning task locally *(can be completed without a GPU)*

Before starting, we recommend running the `simple_example.py` script which provides a concise overview of all three approaches and their key differences.

## Running Individual Examples

Instead of using the `run_all_examples.sh` script, you should run and analyze each example separately to better understand the specific approach, dataset requirements, and educational applications.

### Understanding the Simple Overview Example

**Run command:**
```bash
python simple_example.py
```

This example provides a concise overview of all three fine-tuning approaches without any external dependencies. Use this as your starting point to understand the key differences between methods before diving into the detailed implementations.

### 1. DSPy Prompt Optimization Example

**Run command:**
```bash
python dspy_example.py
```

**Dataset used:** 
- `teacher_student_dialogues.jsonl` - Contains multi-turn student-teacher dialogues showing realistic conversations

**What you'll learn:**
- How prompt optimization works as an alternative to traditional fine-tuning
- How to use Chain-of-Thought reasoning for educational dialogue generation
- Working with smaller datasets (the example uses just 5 dialogues)
- Measuring improvement in pedagogical responses

**Expected results:**
- The example will run in simulation mode if no OpenAI API key is available
- With an API key, it will demonstrate actual prompt optimization
- You'll see before/after comparisons of student dialogue generation quality

**Key educational points:**
- Examine how DSPy structures the optimization process
- Note how the evaluation metrics assess educational quality
- Understand the tradeoff between simplicity and control

### 2. OpenAI Fine-Tuning Example

**Run command:**
```bash
python openai_finetune.py
```

**Datasets used:**
- `openai_teacher_dialogue_training.jsonl` - Training data in OpenAI's chat format
- `openai_teacher_dialogue_validation.jsonl` - Validation data in the same format

**What you'll learn:**
- How to prepare data for OpenAI fine-tuning with the correct format
- The workflow of creating and managing fine-tuning jobs
- Understanding the costs and benefits of cloud-based fine-tuning
- Working with multi-turn conversations and system messages

**Expected results:**
- The example demonstrates the data preparation and job setup process
- It will not actually run a fine-tuning job unless specifically configured
- You'll see the format required for chat completions fine-tuning

**Key educational points:**
- Data formatting is critical - examine the dialogue structure carefully
- Notice the multi-turn nature of educational conversations
- Consider the cost implications of this approach

### 3. HuggingFace LoRA Fine-Tuning Example

**Run command:**
```bash
python huggingface_lora.py
```

**Dataset used:**
- Uses in-memory QA pairs (same content as in `small_edu_qa.jsonl`)
- Format requires input/output pairs for instruction tuning

**What you'll learn:**
- Parameter-efficient fine-tuning with Low-Rank Adaptation (LoRA)
- How to adapt open-source models locally
- Understanding quantization and memory optimization techniques
- Resource requirements for model adaptation

**Expected results:**
- The example will run in simulation mode without a GPU
- It demonstrates the workflow and concepts without actual training
- You'll understand how LoRA reduces the parameter count during fine-tuning

**Key educational points:**
- Examine the LoRA configuration parameters and their impact
- Understand the advantages of parameter-efficient fine-tuning
- Consider when local fine-tuning makes sense for educational applications

## Creating Your Own Educational Datasets

For this assignment, you'll need to create your own datasets for each approach:

1. **For DSPy (Part 1):**
   - Create 5-10 high-quality teacher-student dialogues
   - Focus on realistic conversations with multiple turns
   - Include different pedagogical techniques (Socratic questioning, scaffolding)
   - Save in the same format as `teacher_student_dialogues.jsonl`

2. **For OpenAI (Part 2):**
   - Create 15-20 dialogues in the OpenAI chat format
   - Include a system message defining the teaching approach
   - Structure as alternating user (student) and assistant (teacher) messages
   - Follow the format in `openai_edu_qa_training.jsonl`

3. **For HuggingFace (Part 3):**
   - Create 20+ input/output pairs in a consistent format
   - Ensure clear instruction formatting for each example
   - Include diverse teaching examples and scenarios
   - Use the format shown in the `prepare_dataset()` function

## Detailed Tasks

### Part 1: DSPy Prompt Optimization (3-4 hours)

1. **Setup and Exploration**
   - Run the DSPy example and analyze the outputs
   - Identify where in the code the chain-of-thought reasoning is implemented
   - Examine the `teacher_student_dialogues.jsonl` file format

2. **Dataset Extension**
   - Create 5 additional educational QA pairs on a topic of your choice
   - Add them to the dataset and test their performance

3. **DSPy Module Enhancement**
   - Modify the `EducationalQAModel` to incorporate one of the following:
     - Retrieval-augmented generation
     - Few-shot learning with examples
     - Structured reasoning steps

4. **Evaluation**
   - Develop a more sophisticated evaluation metric than the current simple word matching
   - Test both the original and your enhanced model on the same test set
   - Record performance differences

**Deliverable**: Modified `dspy_enhanced.py` script with your improvements and a short report on performance changes.

### Part 2: OpenAI Fine-Tuning (4-5 hours)

1. **Dataset Preparation**
   - Examine the `openai_edu_qa_training.jsonl` sample file to understand the format
   - Create a specialized dataset of 15-20 QA pairs focused on a single domain (e.g., biology, history)
   - Format this dataset appropriately for OpenAI fine-tuning

2. **Fine-Tuning Setup**
   - Uncomment and modify the fine-tuning section of the provided example
   - Add proper error handling and status monitoring
   - (Optional with instructor approval) Run a small fine-tuning job with OpenAI

3. **Simulation and Testing**
   - If not actually running the fine-tuning job, create a simulation component that mimics expected improvements
   - Design a test set of questions that evaluates domain knowledge in your chosen field

4. **Cost-Benefit Analysis**
   - Calculate the approximate cost of fine-tuning with your dataset
   - Estimate the cost per query for both fine-tuned and non-fine-tuned models
   - Determine at what usage volume fine-tuning becomes cost-effective

**Deliverable**: Your `openai_domain_finetune.py` script, dataset file, and a cost-benefit analysis document.

### Part 3: HuggingFace LoRA Fine-Tuning (6-8 hours)

1. **Environment Setup**
   - Properly configure a Python environment for transformer-based models using the provided conda environment
   - Note that you can complete this part without a GPU by adjusting hyperparameters and model size
   - Study the `small_edu_qa.jsonl` file to understand the data format

2. **Model Selection and Optimization**
   - Choose an appropriate base model from HuggingFace (smaller than the example if needed)
   - For CPU-only training, consider using smaller models like DistilBERT or TinyLlama
   - Experiment with different LoRA hyperparameters (rank, alpha, target modules)
   - Use lower precision (8-bit quantization) and smaller batch sizes to reduce memory requirements
   - Implement training with proper checkpointing and early stopping

3. **Training and Evaluation**
   - Train your LoRA adapter on the educational QA dataset
   - For CPU training, limit the dataset size and number of training steps
   - Evaluate performance before and after adaptation
   - Measure inference speed and memory usage

4. **Adapter Management**
   - Implement proper saving and loading of your LoRA adapter
   - Create a simple inference script that loads only the necessary components

**Note**: If you're working without a GPU, focus on understanding the PEFT/LoRA concepts and code implementation rather than achieving state-of-the-art performance. You can still experiment with and demonstrate the core concepts using smaller models and datasets.

**Deliverable**: Your `huggingface_custom_lora.py` script, trained adapter files, and performance metrics.

## Comparative Analysis (Required for all students)

After completing at least two parts of the assignment, write a 2-3 page analysis comparing the approaches. Address:

1. Development time and effort required
2. Data preparation differences
3. Performance on identical test questions
4. Resource requirements (compute, API costs)
5. Flexibility for adaptation to new domains
6. Practical considerations for deployment

## Bonus Challenges

1. **Hybrid Approach**: Combine DSPy prompt optimization with one of the fine-tuning methods
2. **Extended Evaluation**: Create a benchmark of 50+ questions and evaluate all three approaches
3. **Model Distillation**: Use a fine-tuned model to generate training data for a smaller model
4. **Run All Examples**: Modify the `run_all_examples.sh` script to run your enhanced versions of all three approaches

## Submission Requirements

1. Code files for each part you completed
2. Dataset files created or modified
3. Comparative analysis document
4. Brief reflection on what you learned (1 page)
5. Any trained model artifacts or adapter files

## Evaluation Criteria

- **Functionality** (40%): Does your code work as expected?
- **Understanding** (30%): Does your analysis demonstrate comprehension of the key differences?
- **Innovation** (15%): Have you added meaningful improvements to the base examples?
- **Documentation** (15%): Is your code well-commented and your analysis clear?

## Resources

- **Example Files**:
  - `simple_example.py` - Quick overview of all three approaches
  - `dspy_example.py` - DSPy prompt optimization implementation
  - `openai_finetune.py` - OpenAI fine-tuning implementation
  - `huggingface_lora.py` - HuggingFace LoRA fine-tuning implementation
  - `run_all_examples.sh` - Script to run all three main examples
- **Data Files**:
  - `teacher_student_dialogues.jsonl` - Sample dialogue data for DSPy
  - `small_edu_qa.jsonl` - Sample Q&A pairs for simple examples
  - `openai_edu_qa_training.jsonl` - Sample training data for OpenAI fine-tuning
- **Environment Setup**:
  - `environment.yml` - Located in the project root directory, contains all necessary dependencies for the conda environment
- **All assignment files**: [https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples](https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples)
- **External Documentation**:
  - DSPy documentation: [https://dspy-docs.vercel.app/](https://dspy-docs.vercel.app/)
  - OpenAI fine-tuning guide: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
  - HuggingFace PEFT documentation: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)

---

*Note for instructors: This assignment can be adapted to different technical levels by requiring only certain parts. Part 1 is accessible to most students, while Part 3 requires more advanced ML experience and hardware resources.*
