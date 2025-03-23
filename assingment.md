# LLM Fine-Tuning Methods Assignment: Teacher-Student Interaction Chatbot

## Introduction

In this assignment, you will explore three different approaches to improving LLM performance on a specific task: creating a realistic teacher-student interaction chatbot. You'll work with the provided examples to gain hands-on experience with prompt optimization, cloud-based fine-tuning, and local fine-tuning with parameter-efficient techniques to create a chatbot that can simulate natural educational dialogues rather than just answering questions.

## Learning Objectives

- Understand the key differences between prompt optimization, cloud fine-tuning, and local fine-tuning
- Gain practical experience with DSPy, OpenAI fine-tuning, and HuggingFace LoRA
- Create realistic teacher-student dialogue interactions that go beyond simple Q&A
- Analyze tradeoffs in data requirements, computational resources, and model performance
- Develop skills in evaluating which approach is most suitable for different scenarios

## Prerequisites

- Python 3.8+
- Basic understanding of the three fine-tuning approaches (review `simple_example.py` first)
- Access to the example files: `dspy_example.py`, `openai_finetune.py`, and `huggingface_lora.py`
- For running the examples (optional):
  - DSPy: `pip install dspy-ai==0.11.0`
  - OpenAI: `pip install openai==0.28.0` + API key
  - HuggingFace: `pip install transformers==4.30.0 peft==0.4.0` + GPU

## Getting Started

### 1. Run the basic overview example
Start by running the basic example that explains all three approaches:
```bash
python simple_example.py
```

This will give you a clear understanding of the three fine-tuning methods without requiring any special dependencies.

### 2. Review the code examples
Examine the code for each approach to understand the implementation details:
```bash
cat dspy_example.py
cat openai_finetune.py 
cat huggingface_lora.py
```

Don't worry if you can't run these examples directly - the assignment tasks can be completed by analyzing the code structure and adapting the approaches to your own implementation.

## Assignment Overview

This assignment is structured in three parts of increasing complexity:

1. **Part 1: DSPy Prompt Optimization** - Implement multi-turn teacher-student interactions using DSPy
2. **Part 2: OpenAI Fine-Tuning** - Create and train on realistic teaching dialogues
3. **Part 3: HuggingFace LoRA Fine-Tuning** - Build a local model for educational conversation

## Detailed Tasks

### Part 1: DSPy Prompt Optimization (3-4 hours)

1. **Setup and Exploration**
   - Review the DSPy example code and understand its structure
   - Create a simplified implementation focusing on multi-turn conversation context

2. **Dialogue Dataset Creation**
   - Create 5 educational dialogue sequences that demonstrate effective teacher-student interactions
   - Include examples of Socratic questioning, scaffolding, and misconception handling
   - Format the dialogues appropriately for processing with DSPy

3. **Implementing Your Model**
   - Create a `TeacherStudentDialogueModel` that:
     - Maintains conversation context across multiple turns
     - Applies appropriate teaching strategies based on student questions
     - Provides follow-up questions to check understanding

4. **Evaluation**
   - Develop metrics to evaluate the pedagogical quality of responses
   - Test your implementation with realistic student questions
   - Analyze how well the model maintains context across a conversation

**Deliverable**: A Python script that implements a teacher-student dialogue system using DSPy techniques, even if it doesn't use the exact DSPy Metric class shown in the example.

### Part 2: OpenAI Fine-Tuning (4-5 hours)

1. **Dialogue Dataset Preparation**
   - Create 15-20 teacher-student conversation sequences
   - Include various teaching techniques: clarification requests, scaffolding, worked examples
   - Format this dataset following the structure in openai_finetune.py

2. **Fine-Tuning Setup**
   - Design a fine-tuning workflow based on the example
   - Create a script that would prepare your data for fine-tuning
   - (Optional) If you have API access, implement a small test fine-tuning job

3. **Conversation Testing Plan**
   - Design test scenarios to evaluate a fine-tuned model
   - Create evaluation metrics for teaching quality
   - Develop a plan for implementing the model in an educational setting

4. **Cost-Benefit Analysis**
   - Calculate the approximate cost of fine-tuning with your dialogue dataset
   - Estimate the cost per conversation for both fine-tuned and non-fine-tuned models
   - Determine when fine-tuning becomes cost-effective for educational dialogue applications

**Deliverable**: Your conversation dataset, fine-tuning preparation script, and a cost-benefit analysis document.

### Part 3: HuggingFace LoRA Fine-Tuning (6-8 hours)

1. **Model Selection and Planning**
   - Research and select an appropriate base model for educational dialogues
   - Design the LoRA adaptation strategy based on the example
   - Create a detailed implementation plan

2. **Dataset Preparation**
   - Structure your teacher-student conversation dataset for instruction tuning
   - Include examples of complete teaching sequences with multiple exchanges
   - Format the data following the structure in huggingface_lora.py

3. **Implementation Design**
   - Create a dialogue-specific training setup that preserves conversation context
   - Design an evaluation methodology for teaching quality
   - (Optional) If you have GPU access, implement a small test fine-tuning

4. **Deployment Planning**
   - Design a simple interactive demo that would use your fine-tuned model
   - Create a plan for deploying the model in an educational setting
   - Describe how to implement conversation history management

**Deliverable**: Your dataset preparation code, model adaptation design, and deployment plan document.

## Comparative Analysis (Required for all students)

After completing at least two parts of the assignment, write a 2-3 page analysis comparing the approaches. Address:

1. Effectiveness in maintaining natural educational dialogues
2. Ability to implement various teaching strategies (Socratic, directive, etc.)
3. Data requirements and implementation complexity
4. Practical considerations for classroom deployment
5. Cost-effectiveness and scalability

## Submission Requirements

1. Code files for each part you completed (implementation or detailed pseudocode)
2. Dialogue dataset files you created
3. Comparative analysis document
4. Brief reflection on educational implications (1 page)

## Evaluation Criteria

- **Dialogue Quality** (35%): Does your implementation create natural, pedagogically sound interactions?
- **Technical Understanding** (35%): Does your code/pseudocode demonstrate solid understanding of the approaches?
- **Analysis** (20%): Does your comparative analysis show critical thinking about the tradeoffs?
- **Documentation** (10%): Is your code well-commented and analysis clearly written?

## Additional Resources

- DSPy documentation: https://dspy-docs.vercel.app/
- OpenAI fine-tuning guide: https://platform.openai.com/docs/guides/fine-tuning
- HuggingFace PEFT documentation: https://huggingface.co/docs/peft/index

---

*Note: You don't need to run all examples to complete this assignment. The code review and adaptation to your own implementation is the primary focus.* 