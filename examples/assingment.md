# LLM Fine-Tuning Methods Assignment: Teacher-Student Interaction Chatbot

## Introduction

In this assignment, you will explore three different approaches to improving LLM performance on a specific task: creating a realistic teacher-student interaction chatbot. You'll work with the provided examples to gain hands-on experience with prompt optimization, cloud-based fine-tuning, and local fine-tuning with parameter-efficient techniques to create a chatbot that can simulate natural educational dialogues rather than just answering questions.

All files needed for this assignment are available at: https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples

## Learning Objectives

- Understand the key differences between prompt optimization, cloud fine-tuning, and local fine-tuning
- Gain practical experience with DSPy, OpenAI fine-tuning, and HuggingFace LoRA
- Create realistic teacher-student dialogue interactions that go beyond simple Q&A
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

1. **Part 1: DSPy Prompt Optimization** - Implement multi-turn teacher-student interactions using DSPy
2. **Part 2: OpenAI Fine-Tuning** - Create and train on realistic teaching dialogues
3. **Part 3: HuggingFace LoRA Fine-Tuning** - Build a local model for educational conversation

## Detailed Tasks

### Part 1: DSPy Prompt Optimization (3-4 hours)

1. **Setup and Exploration**
   - Run the DSPy example and analyze the outputs
   - Modify the example to handle multi-turn conversations instead of single Q&A pairs

2. **Dialogue Dataset Creation**
   - Create 5 educational dialogue sequences that demonstrate effective teacher-student interactions
   - Include examples of Socratic questioning, scaffolding, and misconception handling
   - Format the dialogues appropriately for DSPy processing

3. **DSPy Module Enhancement**
   - Modify the `EducationalQAModel` to create `TeacherStudentDialogueModel` that:
     - Maintains conversation context across multiple turns
     - Applies appropriate teaching strategies based on student questions
     - Provides follow-up questions to check understanding

4. **Evaluation**
   - Develop metrics that evaluate the pedagogical quality of responses
   - Test both the original QA model and your enhanced dialogue model
   - Compare how each handles complex educational scenarios

**Deliverable**: Modified `dspy_teacher_dialogue.py` script with your improvements and a short report on the conversational capabilities.

### Part 2: OpenAI Fine-Tuning (4-5 hours)

1. **Dialogue Dataset Preparation**
   - Create 15-20 teacher-student conversation sequences in a single domain
   - Include various teaching techniques: clarification requests, scaffolding, worked examples
   - Format this dataset appropriately for OpenAI fine-tuning (messages format)

2. **Fine-Tuning Setup**
   - Modify the OpenAI example to work with multi-turn conversations
   - Implement proper handling of conversation context and history
   - (Optional with instructor approval) Run a small fine-tuning job with OpenAI

3. **Conversation Testing**
   - Design test scenarios that evaluate the model's ability to:
     - Ask clarifying questions when student queries are vague
     - Provide appropriate levels of assistance based on student understanding
     - Handle misconceptions tactfully
     - Follow up to ensure comprehension

4. **Cost-Benefit Analysis**
   - Calculate the approximate cost of fine-tuning with your dialogue dataset
   - Estimate the cost per conversation for both fine-tuned and non-fine-tuned models
   - Determine when fine-tuning becomes cost-effective for educational dialogue applications

**Deliverable**: Your `openai_teacher_dialogue.py` script, conversation dataset file, and a cost-benefit analysis document.

### Part 3: HuggingFace LoRA Fine-Tuning (6-8 hours)

1. **Environment Setup**
   - Properly configure a Python environment for transformer-based conversational models
   - Verify GPU availability and configuration for LoRA training

2. **Model and Dataset Preparation**
   - Choose an appropriate base model from HuggingFace with good dialogue capabilities
   - Structure your teacher-student conversation dataset for instruction tuning
   - Include examples of complete teaching sequences with multiple exchanges

3. **Training and Evaluation**
   - Implement a dialogue-specific training loop that preserves conversation context
   - Evaluate the model's ability to maintain coherent educational conversations
   - Compare performance on various pedagogical tasks (explanation, questioning, etc.)

4. **Interactive Demo**
   - Create a simple interactive demo that allows testing the dialogue capabilities
   - Implement conversation history management
   - Add ability to switch between different teaching approaches (directive vs. exploratory)

**Deliverable**: Your `huggingface_teacher_dialogue.py` script, trained adapter files, and an interactive demo.

## Comparative Analysis (Required for all students)

After completing at least two parts of the assignment, write a 2-3 page analysis comparing the approaches. Address:

1. Effectiveness in maintaining natural educational dialogues
2. Ability to implement various teaching strategies (Socratic, directive, etc.)
3. Context handling and conversation coherence
4. Response to student confusion or misconceptions
5. Implementation complexity and resource requirements
6. Practical considerations for classroom deployment

## Bonus Challenges

1. **Pedagogical Variety**: Implement multiple teaching styles (Socratic, direct instruction, discovery-based)
2. **Student Modeling**: Add capability to adapt to different student knowledge levels
3. **Subject Specialization**: Fine-tune for a specific educational domain with specialized terminology

## Submission Requirements

1. Code files for each part you completed
2. Dialogue dataset files created
3. Comparative analysis document
4. Brief reflection on educational implications (1 page)
5. Any trained model artifacts or adapter files

## Evaluation Criteria

- **Dialogue Quality** (35%): Does your chatbot create natural, pedagogically sound interactions?
- **Technical Implementation** (35%): Does your code effectively implement the required capabilities?
- **Understanding** (20%): Does your analysis demonstrate comprehension of the key differences?
- **Documentation** (10%): Is your code well-commented and your analysis clear?

## Resources

- Example files in the repository
- All assignment files: https://github.com/UVU-AI-Innovate/UTTA/tree/main/examples
- DSPy documentation: https://dspy-docs.vercel.app/
- OpenAI fine-tuning guide: https://platform.openai.com/docs/guides/fine-tuning
- HuggingFace PEFT documentation: https://huggingface.co/docs/peft/index

---

*Note for instructors: This assignment can be adapted to different technical levels by requiring only certain parts. Part 1 is accessible to most students, while Part 3 requires more advanced ML experience and hardware resources.*
