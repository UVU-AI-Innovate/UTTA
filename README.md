# Utah Teacher Training Assistant (UTTA)

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
conda create -n utta python=3.10
conda activate utta
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Set up model access:

### OpenAI API (Default)
1. Create an OpenAI account at https://platform.openai.com/signup
2. Generate an API key at https://platform.openai.com/api-keys
3. Set your API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

### Alternative Models (Cost-Effective Options)

#### Option 1: Local Ollama Models
1. Install Ollama from https://ollama.ai/
2. Pull and run a model:
```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull Mistral (recommended for teaching tasks)
ollama pull mistral

# For better performance but more resources:
ollama pull llama2
```

3. Update the model in your code:
```python
llm = EnhancedDSPyLLMInterface(model_name="ollama/mistral")
```

#### Option 2: Hugging Face Models
1. Install transformers:
```bash
pip install transformers
```

2. Use open-source models:
```python
from transformers import pipeline

# Phi-2 (Microsoft's compact but powerful model)
model = pipeline('text-generation', model='microsoft/phi-2')

# Or SOLAR (Upstage's efficient model)
model = pipeline('text-generation', model='upstage/SOLAR-10.7B-Instruct-v1.0')
```

#### Cost Comparison (as of 2024):
- GPT-3.5-turbo: $0.0010 / 1K tokens
- GPT-4: $0.03 / 1K tokens
- Local models: Free after initial setup
- Phi-2: Free, 2.7B parameters
- Mistral: Free, 7B parameters
- SOLAR: Free, 10.7B parameters

Choose based on your needs:
- Development/Testing: Use local models (Ollama)
- Production/High accuracy: Use OpenAI API
- Balance: Use Phi-2 or Mistral locally

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

# Initialize the interface (choose your model)
llm = EnhancedDSPyLLMInterface(model_name="gpt-3.5-turbo")  # OpenAI (default)
# llm = EnhancedDSPyLLMInterface(model_name="ollama/mistral")  # Local Mistral
# llm = EnhancedDSPyLLMInterface(model_name="microsoft/phi-2")  # Phi-2

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