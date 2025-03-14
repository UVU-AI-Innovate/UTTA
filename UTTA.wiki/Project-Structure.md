# Project Structure

This page explains the UTTA project structure, organization, and design philosophy. It also identifies potential redundancies and optimization opportunities.

## 📂 Root Directory

The UTTA repository is organized into these main components:

```
UTTA/
├── llm-chatbot-framework/   # Core chatbot functionality
├── llm-fine-tuning/         # Model fine-tuning capabilities
├── benchmark_results/       # Results from benchmark tests
├── evaluation_results/      # Results from evaluation runs
├── offload/                 # Temporary model offloading
├── wiki_backup/             # Wiki backup files
├── UTTA.wiki/               # Wiki documentation
├── .git/                    # Git repository data
├── README.md                # Project overview
└── LICENSE                  # MIT License
```

## 🤖 LLM Chatbot Framework

The chatbot framework (`llm-chatbot-framework/`) is the core implementation for the teaching assistant functionality:

```
llm-chatbot-framework/
├── src/                     # Source code
│   ├── core/                # Core functionality
│   ├── llm/                 # LLM integrations
│   ├── retrieval/           # Knowledge retrieval
│   ├── web/                 # Web interface
│   ├── utils/               # Utility functions
│   └── evaluation/          # Evaluation tools
├── tests/                   # Test suite
├── examples/                # Example implementations
├── knowledge_base/          # Educational content
├── docs/                    # Documentation
├── config/                  # Configuration files
├── data/                    # Data resources
├── cache/                   # Cache storage
├── tools/                   # Development tools
└── config.py                # Main configuration
```

## 🔄 LLM Fine-Tuning Module

The fine-tuning module (`llm-fine-tuning/`) provides tools for customizing LLMs for educational tasks:

```
llm-fine-tuning/
├── src/                     # Source code
│   ├── dspy/                # DSPy fine-tuning
│   ├── huggingface/         # HuggingFace fine-tuning
│   ├── openai/              # OpenAI fine-tuning
│   └── utils/               # Utility functions
├── examples/                # Example fine-tuning
├── finetune.py              # Main entry point
├── compare_approaches.py    # Compare fine-tuning methods
├── check_structure.py       # Validate dataset structure
└── setup.py                 # Package setup
```

## 📊 Results Directories

The project includes directories for storing benchmark and evaluation results:

```
benchmark_results/           # Benchmark performance data
evaluation_results/          # Evaluation metrics and analyses
```

## 🔍 Identified Redundancies

During our review, we identified several potential redundancies in the codebase:

1. **Duplicate Benchmark Directories**:
   - `benchmark_results/` in the root
   - `llm-chatbot-framework/benchmark_results/`
   
   **Recommendation**: Consolidate into a single `benchmark_results/` directory at the root level.

2. **Duplicate Evaluation Directories**:
   - `evaluation_results/` in the root
   - `llm-chatbot-framework/evaluation_results/`
   
   **Recommendation**: Consolidate into a single `evaluation_results/` directory at the root level.

3. **Nested Fine-Tuning Directory**:
   - `llm-fine-tuning/` in the root
   - `llm-chatbot-framework/llm-fine-tuning/`
   
   **Recommendation**: Remove the nested directory and reference the root-level `llm-fine-tuning/` module instead.

4. **File Placement Issues**:
   - `fine_tuning_lesson.html` at the root level
   - `=0.21.0` file at the root level (possibly a fragment from a dependency installation)
   
   **Recommendation**: Move `fine_tuning_lesson.html` to the `llm-fine-tuning/docs/` directory and remove the `=0.21.0` file.

## 🔄 Dependency Management

The project uses multiple requirements files:

- `llm-chatbot-framework/requirements.txt`
- `llm-fine-tuning/requirements.txt`

**Recommendation**: Consider creating a unified requirements management approach with:
- A base `requirements.txt` at the root level
- Component-specific requirements files that use `-r ../requirements.txt` to include the base requirements

## 🌐 Project Interaction Flow

The following diagram illustrates how the different components interact:

```
┌─────────────────────────────────┐
│        llm-fine-tuning          │
│                                 │
│ ┌─────────┐ ┌────────┐ ┌──────┐ │
│ │ DSPy    │ │ OpenAI │ │ HF   │ │
│ │ Tuning  │ │ Tuning │ │ Tune │ │
│ └────┬────┘ └────┬───┘ └───┬──┘ │
│      │           │          │    │
└──────┼───────────┼──────────┼────┘
       │           │          │
       ▼           ▼          ▼
┌─────────────────────────────────┐
│      llm-chatbot-framework      │
│                                 │
│ ┌─────────┐ ┌────────┐ ┌──────┐ │
│ │ Core    │ │ Web UI │ │ Eval │ │
│ │ Engine  │◄┼────────┘ └───┬──┘ │
│ └────┬────┘ │              │    │
│      │      │              │    │
└──────┼──────┼──────────────┼────┘
       │      │              │
       ▼      ▼              ▼
┌─────────┐ ┌─────┐  ┌───────────┐
│ Student │ │ Web │  │Performance│
│ Queries │ │Users│  │ Metrics   │
└─────────┘ └─────┘  └───────────┘
```

## 📝 Development Workflow

The recommended development workflow is:

1. Set up the environment following the [Environment Setup](Environment-Setup) guide
2. Explore example implementations in both main components
3. Implement your educational use case:
   - Prepare datasets using guidance from [Dataset Preparation](Dataset-Preparation)
   - Fine-tune models using the appropriate approach (DSPy, OpenAI, HuggingFace)
   - Integrate the fine-tuned model with the chatbot framework
   - Evaluate performance using the evaluation tools
   - Iterate and improve based on evaluation results

## 🔗 Directory References

- [llm-chatbot-framework/src/](https://github.com/UVU-AI-Innovate/UTTA/tree/main/llm-chatbot-framework/src) - Core implementation
- [llm-fine-tuning/src/](https://github.com/UVU-AI-Innovate/UTTA/tree/main/llm-fine-tuning/src) - Fine-tuning implementation
- [examples/](https://github.com/UVU-AI-Innovate/UTTA/tree/main/llm-chatbot-framework/examples) - Usage examples 