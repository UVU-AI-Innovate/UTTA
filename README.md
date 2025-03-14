# Teacher Training Chatbot Framework

A comprehensive framework for developing and training chatbots specialized in teacher training and pedagogical assistance.

## 🎯 Project Overview

This framework provides a modular system for building chatbots that can assist in teacher training, offering:
- Pedagogical response generation
- Student reaction simulation
- Teaching strategy analysis
- Scaffolded learning support
- Metacognitive prompting

## 🏗️ Project Structure

```
llm-chatbot-framework/
├── src/
│   ├── llm/              # LLM integration and handling
│   │   ├── dspy/        # DSPy-based LLM components
│   │   └── handlers/    # Various LLM handlers
│   ├── knowledge_base/   # Knowledge base integration
│   ├── fine_tuning/      # Model fine-tuning components
│   └── evaluation/       # Evaluation metrics and tools
├── tests/                # Test suites
├── data/                 # Training and evaluation data
└── requirements.txt      # Project dependencies
```

## 🚀 Getting Started

1. Set up the environment:
```bash
conda create -n utta python=3.9
conda activate utta
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Configure your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## 📋 Development Milestones

### 1️⃣ Chatbot Development
- [x] DSPy LLM integration
- [x] Pedagogical language processing
- [x] Student reaction generation
- [x] Teaching analysis

### 2️⃣ Knowledge Base Integration
- [ ] Document indexing
- [ ] Context retrieval
- [ ] Knowledge graph integration
- [ ] Semantic search

### 3️⃣ Fine-Tuning
- [x] Training data preparation
- [x] Model fine-tuning pipeline
- [x] Evaluation metrics
- [ ] Hyperparameter optimization

### 4️⃣ Evaluation
- [x] Automated metrics
- [x] Teaching quality assessment
- [ ] Student engagement analysis
- [ ] Learning outcome evaluation

## 💻 Usage Examples

```python
from src.llm.dspy.handler import EnhancedDSPyLLMInterface

# Initialize the interface
llm = EnhancedDSPyLLMInterface()

# Generate a teaching response
response = llm.generate_comprehensive_response(
    student_query="What is photosynthesis?",
    student_profile={
        "level": "intermediate",
        "learning_style": "visual"
    }
)

# Access different components of the response
print(response['response'])  # Main teaching response
print(response['scaffolding'])  # Scaffolded learning components
print(response['metacognition'])  # Metacognitive prompts
```

## 🧪 Running Tests

```bash
pytest tests/
```

## 📚 Documentation

Each component includes detailed documentation in its respective directory:
- `src/llm/README.md` - LLM integration details
- `src/knowledge_base/README.md` - Knowledge base setup
- `src/fine_tuning/README.md` - Fine-tuning process
- `src/evaluation/README.md` - Evaluation metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.