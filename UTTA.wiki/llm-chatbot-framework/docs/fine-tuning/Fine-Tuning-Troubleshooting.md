# Fine-Tuning Troubleshooting

This page provides solutions to common issues encountered when using the fine-tuning capabilities in the LLM Chatbot Framework.

## General Troubleshooting

### Environment Issues

- **Problem**: Missing dependencies
- **Solution**: 
  ```bash
  # Update all dependencies
  pip install -r requirements.txt
  
  # Run the environment fix script
  ./tools/fix_environment.sh
  ```

### File Path Issues

- **Problem**: File not found errors
- **Solution**: 
  - Ensure all paths are relative to the project root
  - Check file permissions
  - Verify the file exists in the specified location
  - Use absolute paths if necessary

### Logging and Debugging

- **Problem**: Difficulty identifying root cause of issues
- **Solution**:
  - Enable debug logging: `export LOG_LEVEL=DEBUG`
  - Check log files in the project root directory
  - Use the `--verbose` flag with CLI commands when available

## DSPy Optimization Issues

### Module Not Found

- **Problem**: `ModuleNotFoundError` when trying to optimize
- **Solution**:
  - Verify the module name exists in the LLM interface
  - Check that DSPy is properly installed
  - Ensure you're using the correct module name (case-sensitive)

### Optimization Method Error

- **Problem**: Unsupported optimization method
- **Solution**:
  - Use one of the supported methods: `bootstrap_few_shot`, `bootstrap_iterative`
  - Update your DSPy version: `pip install -U dspy-ai`
  - Check compatibility with your LLM provider

### Invalid Training Examples

- **Problem**: Format errors in training examples
- **Solution**:
  - Ensure examples follow the required format:
    ```python
    {
      "input": {"question": "...", "context": "..."},
      "output": {"answer": "..."}
    }
    ```
  - Validate with the CLI: `python tools/fine_tuning_cli.py prepare validate --input examples.json --format dspy`
  - Convert from another format: `python tools/fine_tuning_cli.py prepare convert --input examples.json --output dspy_examples.json --format dspy`

### Save/Load Errors

- **Problem**: Unable to save or load optimized modules
- **Solution**:
  - Check directory permissions
  - Ensure the directory exists
  - Verify module name is consistent between save and load
  - Check for disk space issues

## OpenAI Fine-Tuning Issues

### API Key Errors

- **Problem**: Authentication errors with OpenAI API
- **Solution**:
  - Set the OPENAI_API_KEY environment variable: `export OPENAI_API_KEY=your_key_here`
  - Check API key validity in the OpenAI dashboard
  - Ensure the key has the correct permissions

### File Format Errors

- **Problem**: OpenAI rejects the training file
- **Solution**:
  - Ensure the file is valid JSONL format
  - Each line should be a complete JSON object with "messages" containing "role" and "content"
  - Validate with the CLI: `python tools/fine_tuning_cli.py prepare validate --input data.jsonl --format openai`
  - Fix the format: `python tools/fine_tuning_cli.py prepare convert --input data.json --output data.jsonl --format openai`

### Timeout Errors

- **Problem**: Timeouts when waiting for fine-tuning jobs
- **Solution**:
  - Use asynchronous monitoring instead of waiting: `python tools/fine_tuning_cli.py openai status --job-id your_job_id`
  - For larger jobs, check status periodically rather than waiting
  - Increase timeout settings if applicable

### Cost Management

- **Problem**: Unexpected costs from OpenAI fine-tuning
- **Solution**:
  - Set spending limits in the OpenAI dashboard
  - Monitor usage with the OpenAI dashboard
  - Limit the number of epochs for initial tests
  - Use smaller datasets for initial tests

### Fine-Tuned Model Not Available

- **Problem**: Cannot access fine-tuned model after job completion
- **Solution**:
  - Check job status: `python tools/fine_tuning_cli.py openai status --job-id your_job_id`
  - Wait for the job to fully complete (status: "succeeded")
  - List available models: `python tools/fine_tuning_cli.py openai list-models`
  - Ensure you're using the correct model ID format: `ft:model-name:org:suffix`

## HuggingFace Fine-Tuning Issues

### Memory Errors

- **Problem**: CUDA out of memory errors during fine-tuning
- **Solution**:
  - Use 4-bit quantization: `--load-in-4bit`
  - Reduce batch size: `--batch-size 1`
  - Enable gradient accumulation: `--gradient-accumulation-steps 4`
  - Use a smaller model
  - Free up GPU memory from other processes

### CUDA Errors

- **Problem**: CUDA not available or CUDA version issues
- **Solution**:
  - Check CUDA installation: `nvidia-smi`
  - Ensure PyTorch CUDA version matches installed CUDA: `python -c "import torch; print(torch.version.cuda)"`
  - For incompatible CUDA, try CPU-only fine-tuning (very slow)
  - Install the correct PyTorch version: `pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html`

### Model Loading Errors

- **Problem**: Cannot load HuggingFace models
- **Solution**:
  - Check if you have sufficient disk space
  - Ensure internet connection for downloading models
  - Verify you have access to the model (for gated models like Llama)
  - Set HuggingFace token for gated models: `export HUGGINGFACE_TOKEN=your_token`
  - Try loading with fewer features: `--low_cpu_mem_usage`

### Training Stuck

- **Problem**: Training seems stuck or progress is very slow
- **Solution**:
  - Check GPU utilization: `nvidia-smi -l 1`
  - Reduce model size or use quantization
  - Check for infinite loops in training code
  - Look for errors in the training logs
  - Increase logging verbosity to track progress

### Adapter Integration Issues

- **Problem**: Issues with LoRA adapters after fine-tuning
- **Solution**:
  - Verify adapter files exist in the output directory
  - Check that the model and adapter architectures match
  - Ensure you're using compatible PEFT versions: `pip install -U peft`
  - Try exporting the model: `python tools/fine_tuning_cli.py huggingface export-model --model-dir your_model_dir`

## CLI Tool Issues

### Command Not Found

- **Problem**: CLI tool command not found
- **Solution**:
  - Make sure you're in the project root directory
  - Check that the file is executable: `chmod +x tools/fine_tuning_cli.py`
  - Run with python explicitly: `python tools/fine_tuning_cli.py`

### Subcommand Errors

- **Problem**: Invalid subcommand or arguments
- **Solution**:
  - Check command syntax: `python tools/fine_tuning_cli.py --help`
  - Check subcommand help: `python tools/fine_tuning_cli.py dspy --help`
  - Ensure all required arguments are provided
  - Check for typos in command names

### Data Preparation Errors

- **Problem**: Data conversion or validation fails
- **Solution**:
  - Check input file format
  - Ensure the output format is one of: `openai`, `dspy`, `huggingface`
  - Fix JSON formatting issues manually
  - For complex data, write a custom conversion script

## Related Pages

- [Fine-Tuning Overview](Fine-Tuning-Overview)
- [DSPy Optimization](DSPy-Optimization)
- [OpenAI Fine-Tuning](OpenAI-Fine-Tuning)
- [HuggingFace Fine-Tuning](HuggingFace-Fine-Tuning)
- [Fine-Tuning CLI](Fine-Tuning-CLI)
- [Fine-Tuning Best Practices](Fine-Tuning-Best-Practices) 