# UTTA Data

This directory contains data files used by the UTTA framework.

## Directory Structure

- **evaluation/**: Contains evaluation results and metrics
- **benchmarks/**: Contains benchmark results and performance data
- **knowledge_base/**: Sample knowledge bases used for demonstrations

## Usage

Data files in this directory are used by various UTTA components, including:

- The vector database for semantic search
- Training examples for fine-tuning
- Evaluation datasets
- Benchmark configurations

To use these data files in your code:

```python
from pathlib import Path

# Get the data directory
data_dir = Path(__file__).parent.parent / "data"

# Access specific files
knowledge_file = data_dir / "knowledge_base" / "sample_knowledge.json"
``` 