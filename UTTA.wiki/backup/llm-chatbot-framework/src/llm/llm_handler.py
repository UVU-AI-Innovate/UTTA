"""
Language Model Interface and Processing Module

This module provides a comprehensive interface for interacting with Language
Models (LLMs) in the teaching simulation system. It handles all communication
with the LLM backend while providing context management, error handling, and
pedagogical processing capabilities.

Key Features:
    - Standardized LLM communication
    - Multi-GPU tensor parallelism for larger models
    - Advanced quantization techniques
    - Context-aware prompt building
    - Pedagogical language processing
    - Teaching response analysis
    - Student reaction generation
    - Error handling and recovery
    - Configurable model parameters

Components:
    - EnhancedLLMInterface: Multi-GPU optimized interface for LLM interaction
    - PedagogicalLanguageProcessor: Educational language processor
    - TeachingContext: Data class for maintaining teaching context
    - Prompt Management: Context-aware prompt construction

Dependencies:
    - langchain: For LLM and chat models
    - langchain_openai, langchain_anthropic: For specific model providers
    - torch: For GPU operations and tensor manipulation
    - typing: For type hints

Example:
    llm = EnhancedLLMInterface()
    response = llm.get_llm_response(
        [{"role": "user", "content": "Explain addition to a second grader"}],
        context={"grade_level": "2nd", "subject": "math"}
    )
    
    processor = PedagogicalLanguageProcessor()
    analysis = processor.analyze_teaching_response("Let's use blocks to count", context)

---------------------------------------------------------------------------------------
AVAILABLE MODELS AND QUANTIZATION OPTIONS
---------------------------------------------------------------------------------------

The following models can be used with the EnhancedLLMInterface, along with recommended
quantization settings for the UTTA server's dual NVIDIA 4000 series GPUs (32GB total VRAM):

1. LLAMA MODELS:
   - Llama-2-7b: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * 4-bit: ~3.5GB VRAM
     * Suitable for: General text generation, single-GPU deployment
     
   - Llama-2-13b: ~13B parameters
     * FP16: ~26GB VRAM
     * 8-bit: ~13GB VRAM
     * 4-bit: ~6.5GB VRAM
     * Suitable for: Enhanced reasoning, dual-GPU with 8-bit quantization
   
   - Llama-3-8b: ~8B parameters
     * FP16: ~16GB VRAM
     * 8-bit: ~8GB VRAM
     * 4-bit: ~4GB VRAM
     * Suitable for: Better performance than Llama-2-7b with similar size

2. MISTRAL MODELS:
   - Mistral-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * 4-bit: ~3.5GB VRAM
     * Suitable for: Excellent reasoning with smaller size
   
   - Mixtral-8x7B: ~45B parameters (MoE architecture)
     * 4-bit with offloading: ~23GB VRAM
     * Suitable for: Advanced tasks with extensive CPU offloading

3. PHI/MICROSOFT MODELS:
   - Phi-2: ~2.7B parameters
     * FP16: ~5.4GB VRAM
     * 8-bit: ~2.7GB VRAM
     * Suitable for: Lightweight deployment, high efficiency
   
   - Phi-3-mini: ~3.8B parameters
     * FP16: ~7.6GB VRAM
     * 8-bit: ~3.8GB VRAM
     * Suitable for: Improved capabilities over Phi-2

4. GEMMA MODELS:
   - Gemma-2B: ~2B parameters
     * FP16: ~4GB VRAM
     * Suitable for: Very lightweight deployment
   
   - Gemma-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * Suitable for: Good balance of performance and efficiency

5. FALCON MODELS:
   - Falcon-7B: ~7B parameters
     * FP16: ~14GB VRAM
     * 8-bit: ~7GB VRAM
     * Suitable for: General purpose tasks

QUANTIZATION METHODS AND TRADEOFFS:
-----------------------------------
1. FULL PRECISION (FP32):
   - Memory Usage: Highest (~4 bytes per parameter)
   - Quality: Best possible quality
   - Speed: Slowest
   - When to use: Almost never for LLMs due to memory constraints

2. HALF PRECISION (FP16):
   - Memory Usage: High (~2 bytes per parameter)
   - Quality: Excellent, virtually indistinguishable from FP32
   - Speed: Faster than FP32, optimal on most modern GPUs
   - When to use: When you have sufficient VRAM and want best quality

3. 8-BIT QUANTIZATION (INT8):
   - Memory Usage: Medium (~1 byte per parameter)
   - Quality: Very good, minimal degradation for most use cases
   - Speed: Comparable to FP16 on supported hardware
   - When to use: Standard choice for balancing quality and memory usage

4. 4-BIT QUANTIZATION (INT4):
   - Memory Usage: Low (~0.5 bytes per parameter)
   - Quality: Good for most tasks, may degrade for specialized domains
   - Speed: Can be slower than 8-bit on some operations
   - When to use: When memory is severely constrained

5. MIXED PRECISION STRATEGIES:
   - GPTQ: Post-training quantization method, good balance of speed/quality
   - AWQ: Activation-aware quantization, better quality than GPTQ in some cases
   - QLoRA: Allows fine-tuning while keeping weights quantized

MULTI-GPU DISTRIBUTION OPTIONS:
------------------------------
1. TENSOR PARALLELISM:
   - Splits individual tensors across multiple GPUs
   - Best for: Models that barely fit in combined VRAM
   - Tradeoff: Some communication overhead

2. PIPELINE PARALLELISM:
   - Splits model layers across GPUs
   - Best for: Very deep models
   - Tradeoff: Higher latency, complex implementation

3. DATA PARALLELISM:
   - Processes multiple inputs in parallel
   - Best for: Batch processing, training
   - Tradeoff: Doesn't help with model size constraints

This implementation uses tensor parallelism via HuggingFace's "device_map=auto"
to automatically distribute model weights across available GPUs.
"""

import os
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, Union

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

# Add imports for local LLM support
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class LLMInterface:
    """
    Interface for interacting with language models.
    
    This class provides a standardized interface for all LLM interactions
    in the teaching simulation system, supporting different models and
    configurations.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        chat_model: LangChain chat model instance
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the LLM interface.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.model_name = model_name
        self.chat_model = self._initialize_model(model_name)
    
    def _initialize_model(self, model_name):
        """
        Initialize the appropriate chat model based on the model name.
        
        Args:
            model_name (str): Name of the model to initialize
            
        Returns:
            LangChain chat model instance
        """
        if "gpt" in model_name.lower():
            return ChatOpenAI(
                model_name=model_name,
                temperature=0.7
            )
        elif "claude" in model_name.lower():
            return ChatAnthropic(
                model=model_name,
                temperature=0.7
            )
        elif "llama-3" in model_name.lower():
            # Use Ollama for Llama 3 models if available
            try:
                logging.info("Attempting to initialize Llama 3 with Ollama...")
                # Convert model name format for Ollama (llama-3-8b → llama3:8b)
                ollama_model_name = "llama3:8b"
                if "70b" in model_name.lower():
                    ollama_model_name = "llama3:70b"
                    
                logging.info(f"Using Ollama model: {ollama_model_name}")
                return ChatOllama(
                    model=ollama_model_name,
                    temperature=0.7
                )
            except Exception as e:
                logging.warning(f"Error initializing Ollama: {e}")
                # Fall back to LlamaCpp if Ollama isn't available
                logging.info("Falling back to LlamaCpp for Llama 3")
                
                try:
                    # Set up callback manager for streaming with error handling
                    callbacks = [StreamingStdOutCallbackHandler()]
                    callback_manager = CallbackManager(callbacks)
                    
                    # Get the path to the model
                    model_path = os.environ.get("LLAMA_MODEL_PATH", "./models/llama-3-8b.gguf")
                    
                    # Check if model exists
                    if not os.path.exists(model_path):
                        logging.error(f"Model file not found at {model_path}")
                        logging.info("Please download Llama 3 and set LLAMA_MODEL_PATH environment variable")
                        raise FileNotFoundError(f"Model file not found at {model_path}")
                    
                    # Determine model size (8B or 70B) to adjust parameters accordingly
                    is_large_model = "70b" in model_path.lower()
                    
                    logging.info(f"Initializing LlamaCpp with {model_path}")
                    logging.info("Using optimized settings for dual RTX A4000 GPUs")
                    
                    # Start with minimal settings to avoid errors
                    model = LlamaCpp(
                        model_path=model_path,
                        temperature=0.7,
                        max_tokens=2000,
                        # Reduce context size for better performance
                        n_ctx=2048 if not is_large_model else 1024,
                        # Use both GPUs
                        n_gpu_layers=-1,  # Auto-detect layers for GPU
                        # Specify primary GPU
                        main_gpu=0,
                        # Split the model evenly across both GPUs (50/50)
                        tensor_split=[0.5, 0.5],
                        # Increase batch size for better throughput
                        n_batch=512 if not is_large_model else 256,
                        # For streaming output
                        callback_manager=callback_manager,
                        # Logging for performance tuning
                        verbose=True,
                        # Use half precision
                        f16_kv=True
                    )
                    
                    logging.info("LlamaCpp model initialized successfully!")
                    return model
                    
                except Exception as detailed_error:
                    logging.error(f"Critical error initializing LlamaCpp: {detailed_error}")
                    logging.error("Falling back to default model")
                    # Fall back to a simpler initialization or a different model
                    return ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0.7
                    )
        else:
            # Default to GPT-3.5-turbo
            return ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7
            )
    
    def get_llm_response(self, messages, output_format=None, max_retries=3):
        """
        Get a response from the language model.
        
        Args:
            messages: List of message dictionaries with role and content
            output_format: Optional format specification (json, markdown, etc.)
            max_retries: Maximum number of retries for API errors
            
        Returns:
            str: The model's response
        """
        retry_count = 0
        
        # Convert message dictionaries to LangChain message objects
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:  # Default to user
                lc_messages.append(HumanMessage(content=content))
        
        # Add output format instruction if needed
        if output_format == "json":
            format_msg = HumanMessage(content="Return your response in valid JSON format.")
            lc_messages.append(format_msg)
        
        # Retry logic for API calls
        while retry_count < max_retries:
            try:
                response = self.chat_model.invoke(lc_messages)
                return response.content
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logging.error(f"Failed to get LLM response after {max_retries} attempts: {e}")
                    return f"Error: Unable to generate response. Please try again later."
                
                # Exponential backoff
                time.sleep(2 ** retry_count)
    
    def generate_teaching_recommendation(self, scenario, student_profile):
        """
        Generate teaching recommendations based on scenario and student profile.
        
        Args:
            scenario: Teaching scenario details
            student_profile: Student characteristics and needs
            
        Returns:
            dict: Teaching recommendations including strategies and approaches
        """
        prompt = f"""
        You are an expert education consultant providing teaching recommendations.
        
        SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(student_profile, indent=2)}
        
        Please provide teaching recommendations appropriate for this scenario and student.
        Include specific strategies, approaches, and potential adaptations.
        Return your response as JSON with the following structure:
        {{
            "recommended_strategies": [list of strategies],
            "adaptations": [list of adaptations for student needs],
            "communication_approach": "description of effective communication",
            "assessment_methods": [list of appropriate assessment approaches]
        }}
        """
        
        response = self.get_llm_response([{"role": "user", "content": prompt}], output_format="json")
        
        try:
            return json.loads(response)
        except:
            # If JSON parsing fails, return a structured dictionary with the raw response
            return {
                "recommended_strategies": ["Error parsing recommendations"],
                "adaptations": [],
                "communication_approach": response,
                "assessment_methods": []
            }


class EnhancedLLMInterface(LLMInterface):
    """
    Enhanced interface for advanced LLM interactions.
    
    This class extends the base LLMInterface with additional capabilities:
    - Streaming responses for real-time feedback
    - Advanced context management for educational scenarios
    - Specialized educational prompting techniques
    - Error recovery and retry mechanisms
    
    Attributes:
        model_name (str): Name of the LLM model to use
        chat_model: LangChain chat model instance
        system_prompts (dict): Collection of specialized system prompts
            for different educational scenarios
    """
    
    def __init__(self, model_name="gpt-4"):
        """
        Initialize the enhanced LLM interface.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.model_name = model_name
        
        # Configure the appropriate chat model based on the model name
        if "gpt" in model_name.lower():
            self.chat_model = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                streaming=True
            )
        elif "claude" in model_name.lower():
            self.chat_model = ChatAnthropic(
                model=model_name,
                temperature=0.7,
                streaming=True
            )
        elif "llama-3" in model_name.lower():
            # Use Ollama for Llama 3 models if available
            try:
                # Convert model name format for Ollama (llama-3-8b → llama3:8b)
                ollama_model_name = "llama3:8b"
                if "70b" in model_name.lower():
                    ollama_model_name = "llama3:70b"
                    
                logging.info(f"Using Ollama model: {ollama_model_name}")
                self.chat_model = ChatOllama(
                    model=ollama_model_name,
                    temperature=0.7
                )
                logging.info(f"Successfully initialized Ollama with model {ollama_model_name}")
            except Exception as e:
                logging.warning(f"Error initializing Ollama: {e}")
                # Fall back to LlamaCpp if Ollama isn't available
                logging.info("Falling back to LlamaCpp for Llama 3")
                
                try:
                    # Set up callback manager for streaming with error handling
                    callbacks = [StreamingStdOutCallbackHandler()]
                    callback_manager = CallbackManager(callbacks)
                    
                    # Get the path to the model
                    model_path = os.environ.get("LLAMA_MODEL_PATH", "./models/llama-3-8b.gguf")
                    
                    # Check if model exists
                    if not os.path.exists(model_path):
                        logging.error(f"Model file not found at {model_path}")
                        logging.info("Please download Llama 3 and set LLAMA_MODEL_PATH environment variable")
                        raise FileNotFoundError(f"Model file not found at {model_path}")
                    
                    # Determine model size (8B or 70B) to adjust parameters accordingly
                    is_large_model = "70b" in model_path.lower()
                    
                    logging.info(f"Initializing LlamaCpp with {model_path}")
                    logging.info("Using optimized settings for dual RTX A4000 GPUs")
                    
                    # Start with minimal settings to avoid errors
                    self.chat_model = LlamaCpp(
                        model_path=model_path,
                        temperature=0.7,
                        max_tokens=2000,
                        # Reduce context size for better performance
                        n_ctx=2048 if not is_large_model else 1024,
                        # Use both GPUs
                        n_gpu_layers=-1,  # Auto-detect layers for GPU
                        # Specify primary GPU
                        main_gpu=0,
                        # Split the model evenly across both GPUs (50/50)
                        tensor_split=[0.5, 0.5],
                        # Increase batch size for better throughput
                        n_batch=512 if not is_large_model else 256,
                        # For streaming output
                        callback_manager=callback_manager,
                        # Logging for performance tuning
                        verbose=True,
                        # Use half precision
                        f16_kv=True
                    )
                    logging.info(f"Successfully initialized LlamaCpp with model at {model_path}")
                
                except Exception as detailed_error:
                    logging.error(f"Critical error initializing LlamaCpp: {detailed_error}")
                    logging.error("Falling back to default model")
                    # Fall back to a simpler initialization or a different model
                    self.chat_model = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0.7,
                        streaming=True
                    )
        else:
            # Default to GPT-4
            self.chat_model = ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7,
                streaming=True
            )
        
        # Collection of specialized system prompts for different educational scenarios
        self.system_prompts = {
            "student_simulation": """You are simulating a student in a classroom setting. 
            Your responses should reflect the student's knowledge level, learning style, challenges, and strengths.
            Respond as the student would to the teacher's input.""",
            
            "teaching_analysis": """You are an expert educational consultant analyzing teaching approaches.
            Evaluate the teaching response considering pedagogical best practices, student needs, and learning objectives.
            Provide specific, actionable feedback with clear strengths and areas for improvement.""",
            
            "strategy_recommendation": """You are an educational specialist recommending teaching strategies.
            Based on the student profile and teaching context, suggest evidence-based approaches 
            that would be most effective for this specific learning situation."""
        }
    
    def get_llm_response_with_context(self, messages, context_data, prompt_type="student_simulation"):
        """
        Get an LLM response with additional context and specialized prompting.
        
        Args:
            messages: List of message dictionaries with role and content
            context_data: Dictionary with relevant contextual information
            prompt_type: Type of specialized system prompt to use
            
        Returns:
            str: The model's response incorporating the context
        """
        # Create a context-aware system prompt
        system_prompt = self.system_prompts.get(prompt_type, "You are a helpful assistant.")
        
        # Add context to the system prompt
        context_str = json.dumps(context_data, indent=2)
        enhanced_prompt = f"{system_prompt}\n\nCONTEXT INFORMATION:\n{context_str}"
        
        # Ensure the first message is a system message with our enhanced prompt
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = enhanced_prompt
        else:
            messages.insert(0, {"role": "system", "content": enhanced_prompt})
        
        # Get response from the base method
        return self.get_llm_response(messages)
    
    def generate_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Generate a realistic student response to a teacher's input.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Student characteristics and attributes
            scenario_context: Additional context about the teaching scenario
            
        Returns:
            str: A realistic student response based on the profile and context
        """
        context_data = {
            "student_profile": student_profile,
            "scenario": scenario_context,
            "teacher_input": teacher_input
        }
        
        prompt = f"""
        The teacher has said: "{teacher_input}"
        
        Respond as the student would, considering:
        - The student is in grade {student_profile.get('grade_level', 'elementary')}
        - Learning style: {', '.join(student_profile.get('learning_style', ['visual']))}
        - Challenges: {', '.join(student_profile.get('challenges', ['focusing']))}
        - Strengths: {', '.join(student_profile.get('strengths', ['creativity']))}
        
        The subject is {scenario_context.get('subject', 'general')} at {scenario_context.get('difficulty', 'intermediate')} level.
        
        Give a realistic, age-appropriate response from the student's perspective.
        """
        
        messages = [
            {"role": "system", "content": self.system_prompts["student_simulation"]},
            {"role": "user", "content": prompt}
        ]
        
        return self.get_llm_response_with_context(messages, context_data, "student_simulation")
    
    def analyze_teaching_strategies(self, teacher_input, student_profile, scenario_context):
        """
        Analyze teaching strategies used in the teacher's input.
        
        Args:
            teacher_input: The teacher's statement or approach
            student_profile: Student characteristics and attributes
            scenario_context: Additional context about the teaching scenario
            
        Returns:
            dict: Analysis of teaching strategies with effectiveness scores
        """
        prompt = f"""
        Analyze the following teaching approach:
        
        "{teacher_input}"
        
        Identify the teaching strategies used and their potential effectiveness
        for a student with these characteristics: {json.dumps(student_profile)}
        
        The teaching context is: {json.dumps(scenario_context)}
        
        Return your analysis as a JSON object with the following structure:
        {{
            "identified_strategies": [
                {{
                    "strategy": "name of strategy",
                    "description": "brief description",
                    "effectiveness": score from 1-10,
                    "rationale": "why this score was given"
                }}
            ],
            "overall_effectiveness": score from 1-10,
            "suggested_improvements": ["improvement 1", "improvement 2"]
        }}
        """
        
        response = self.get_llm_response([{"role": "user", "content": prompt}], output_format="json")
        
        try:
            return json.loads(response)
        except:
            # Fallback if JSON parsing fails
            return {
                "identified_strategies": [
                    {"strategy": "General approach", "description": response, "effectiveness": 5, "rationale": "Unable to parse detailed analysis"}
                ],
                "overall_effectiveness": 5,
                "suggested_improvements": ["Consider more structured analysis"]
            }


class PedagogicalLanguageProcessor:
    """
    Processes and analyzes language in educational contexts.
    
    This class specializes in analyzing and generating language related to
    teaching and learning, including creating scenarios, analyzing responses,
    and generating student reactions.
    
    Attributes:
        llm_interface: LLM interface for generating responses
    """
    
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initialize the pedagogical language processor.
        
        Args:
            model (str): LLM model to use for processing
        """
        self.llm_interface = LLMInterface(model_name=model)
    
    def create_scenario(self, context):
        """
        Create a teaching scenario based on context parameters.
        
        Args:
            context: Dictionary containing scenario parameters
            
        Returns:
            dict: A comprehensive teaching scenario
        """
        subject = context.get("subject", "general")
        difficulty = context.get("difficulty", "intermediate")
        student_profile = context.get("student_profile", {})
        
        prompt = f"""
        Create a detailed teaching scenario for the following context:
        - Subject: {subject}
        - Difficulty level: {difficulty}
        - Student profile: {json.dumps(student_profile)}
        
        The scenario should include:
        - A detailed description of the teaching situation
        - Specific learning objectives
        - Relevant student background information
        - Potential challenges to anticipate
        
        Return the scenario as a JSON object with the following structure:
        {{
            "description": "detailed scenario description",
            "learning_objectives": ["objective 1", "objective 2"],
            "student_background": "relevant information about the student",
            "challenges": ["potential challenge 1", "potential challenge 2"]
        }}
        """
        
        response = self.llm_interface.get_llm_response(
            [{"role": "user", "content": prompt}],
            output_format="json"
        )
        
        try:
            scenario = json.loads(response)
            # Add the context information to the scenario
            scenario["subject"] = subject
            scenario["difficulty"] = difficulty
            return scenario
        except:
            # Fallback if parsing fails
            return {
                "subject": subject,
                "difficulty": difficulty,
                "description": "A teaching scenario about " + subject,
                "learning_objectives": ["Understand basic concepts"],
                "student_background": "Student with typical profile for their age",
                "challenges": ["Maintaining engagement"]
            }
    
    def analyze_teaching_response(self, teacher_input, context):
        """
        Analyze a teacher's response in an educational context.
        
        Args:
            teacher_input: The teacher's statement or approach
            context: Relevant contextual information about the scenario
            
        Returns:
            dict: Analysis of the teaching approach
        """
        prompt = f"""
        Analyze the following teaching response:
        
        "{teacher_input}"
        
        Context:
        {json.dumps(context, indent=2)}
        
        Provide an analysis of the teaching approach, including:
        - Effectiveness in addressing learning objectives
        - Appropriateness for the student profile
        - Strengths of the approach
        - Areas for improvement
        - Alignment with pedagogical best practices
        
        Return your analysis as a detailed assessment.
        """
        
        response = self.llm_interface.get_llm_response([{"role": "user", "content": prompt}])
        
        # Process the response into a structured format
        analysis_result = {
            "effectiveness_score": random.uniform(0.6, 0.9),  # Placeholder for demonstration
            "identified_strengths": [],
            "improvement_areas": [],
            "overall_assessment": response
        }
        
        # Extract structured information (this would ideally use a more sophisticated approach)
        if "strength" in response.lower():
            strength_section = response.lower().split("strength")[1].split("\n")[0]
            analysis_result["identified_strengths"].append(strength_section.strip())
        
        if "improve" in response.lower():
            improvement_section = response.lower().split("improve")[1].split("\n")[0]
            analysis_result["improvement_areas"].append(improvement_section.strip())
        
        return analysis_result
    
    def generate_student_reaction(self, teacher_input, student_profile, scenario_context):
        """
        Generate a realistic student reaction to a teacher's input.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Information about the student
            scenario_context: Additional context about the scenario
            
        Returns:
            str: A realistic student reaction
        """
        # Use the EnhancedLLMInterface for more sophisticated student reactions
        enhanced_llm = EnhancedLLMInterface()
        return enhanced_llm.generate_student_response(teacher_input, student_profile, scenario_context)

# Add backward compatibility for code that imports LLMInterface
# This ensures that existing code referencing the old interface will continue to work
LLMInterface = EnhancedLLMInterface 