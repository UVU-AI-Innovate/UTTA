# Fine-tuning LLMs for Educational Applications

This guide explains three methods for fine-tuning Large Language Models to simulate a second-grade student chatbot.

## Learning Objectives

- Compare three fine-tuning approaches: DSPy, OpenAI, and LoRA
- Create effective datasets for each method
- Choose the appropriate method for your use case
- Implement and evaluate your fine-tuned model

## Method Comparison

| Method | Description | Data Needed | Best For |
|--------|-------------|-------------|----------|
| **DSPy** | Optimizes prompts without changing model weights | 10-50 examples | Quick prototyping, limited data |
| **OpenAI** | Updates model weights through cloud API | 50-100+ examples | Production chatbots, consistent behavior |
| **LoRA** | Adapts specific model parameters locally | 100+ examples | Complete control, data privacy |

## Dataset Formats

### DSPy Format
```json
{
  "dialogue": [
    {"role": "teacher", "content": "What happens when ice melts?"},
    {"role": "student", "content": "It turns into water! We did an experiment in class."},
    {"role": "teacher", "content": "Why do you think it melted?"},
    {"role": "student", "content": "Because the sun is hot!"}
  ]
}
```

### OpenAI Format
```json
{
  "messages": [
    {"role": "system", "content": "You are a second-grade student."},
    {"role": "user", "content": "What is the water cycle?"},
    {"role": "assistant", "content": "Water goes up to the sky and makes clouds."}
  ]
}
```

### LoRA Format
```json
{
  "question": "How do plants grow?",
  "answer": "Plants grow from seeds! First you put a seed in dirt and water it."
}
```

## Implementation Steps

1. **Setup Environment**
   - Python 3.10+, Conda, OpenAI API key (for DSPy/OpenAI)
   - Install dependencies: `conda env create -f environment.yml`

2. **Create Dataset**
   - Follow the appropriate format for your chosen method
   - Include age-appropriate content (vocabulary level: 2,000-5,000 words)
   - Cover various subjects: math, science, language arts, social studies

3. **Implement Your Chosen Method**
   - **DSPy**: Create dialogue dataset → Define signatures → Run optimization
   - **OpenAI**: Format messages → Upload dataset → Initiate fine-tuning
   - **LoRA**: Prepare QA pairs → Configure parameters → Train locally

4. **Evaluation**
   - Test with various subject-matter questions
   - Verify age-appropriate responses and reasoning
   - Check for natural conversation flow and educational value

## Repository Structure

```
examples/
├── README.md                                 # This file
├── Approach 1: DSPy Prompt Optimization
│   ├── dspy_example.py
│   └── teacher_student_dialogues.jsonl
├── Approach 2: OpenAI Fine-tuning
│   ├── openai_finetune.py
│   └── openai_teacher_dialogue_training.jsonl
└── Approach 3: HuggingFace LoRA
    ├── huggingface_lora.py
    └── small_edu_qa.jsonl
```

## Resources

- DSPy: [https://dspy.ai/](https://dspy.ai/)
- OpenAI Fine-tuning: [https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- HuggingFace LoRA: [https://huggingface.co/docs/peft/conceptual_guides/lora](https://huggingface.co/docs/peft/conceptual_guides/lora)
