# LLM Fine-Tuning Methods Assignment

## Introduction

In this assignment, you will explore three different approaches to improving LLM performance on a specific task: educational question answering. You'll work with the provided examples to gain hands-on experience with prompt optimization, cloud-based fine-tuning, and local fine-tuning with parameter-efficient techniques.

## Learning Objectives

- Understand the key differences between prompt optimization, cloud fine-tuning, and local fine-tuning
- Gain practical experience with DSPy, OpenAI fine-tuning, and HuggingFace LoRA
- Analyze tradeoffs in data requirements, computational resources, and model performance
- Develop skills in evaluating which approach is most suitable for different scenarios

## Prerequisites

- Python 3.8+
- Access to the example files: `dspy_example.py`, `openai_finetune.py`, and `huggingface_lora.py`
- OpenAI API key (for Parts 1 and 2)
- CUDA-compatible GPU with 8GB+ VRAM (for Part 3, optional)
- Installed requirements from `requirements.txt`

## Assignment Overview

This assignment is structured in three parts of increasing complexity:

1. **Part 1: DSPy Prompt Optimization** - Modify and extend the DSPy example
2. **Part 2: OpenAI Fine-Tuning** - Prepare data and set up an OpenAI fine-tuning job
3. **Part 3: HuggingFace LoRA Fine-Tuning** - Run a parameter-efficient fine-tuning task locally

## Detailed Tasks

### Part 1: DSPy Prompt Optimization (3-4 hours)

1. **Setup and Exploration**
   - Run the DSPy example and analyze the outputs
   - Identify where in the code the chain-of-thought reasoning is implemented

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
   - Properly configure a Python environment for transformer-based models
   - Verify GPU availability and configuration for LoRA training

2. **Model Selection and Optimization**
   - Choose an appropriate base model from HuggingFace (smaller than the example if needed)
   - Experiment with different LoRA hyperparameters (rank, alpha, target modules)
   - Implement training with proper checkpointing and early stopping

3. **Training and Evaluation**
   - Train your LoRA adapter on the educational QA dataset
   - Evaluate performance before and after adaptation
   - Measure inference speed and memory usage

4. **Adapter Management**
   - Implement proper saving and loading of your LoRA adapter
   - Create a simple inference script that loads only the necessary components

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

- Example files in the repository
- DSPy documentation: https://dspy-docs.vercel.app/
- OpenAI fine-tuning guide: https://platform.openai.com/docs/guides/fine-tuning
- HuggingFace PEFT documentation: https://huggingface.co/docs/peft/index

---

*Note for instructors: This assignment can be adapted to different technical levels by requiring only certain parts. Part 1 is accessible to most students, while Part 3 requires more advanced ML experience and hardware resources.*
