"""
System metrics monitoring for LLM evaluation.
"""

from typing import Dict, Any, Optional
import psutil
import torch
import GPUtil
from dataclasses import dataclass
from transformers import AutoModel
import sys
import os
from pathlib import Path

@dataclass
class GPUStats:
    """GPU statistics."""
    memory_used: float  # in MB
    memory_total: float  # in MB
    utilization: float  # in %
    temperature: float  # in Celsius

@dataclass
class ModelStats:
    """Model statistics."""
    model_name: str
    model_size: float  # in MB
    quantization: Optional[str]
    precision: str
    device: str
    num_parameters: int

class SystemMetrics:
    """Monitor system metrics during LLM operations."""
    
    def __init__(self):
        """Initialize system metrics monitor."""
        self.has_gpu = torch.cuda.is_available()
        
    def get_gpu_stats(self) -> Optional[Dict[int, GPUStats]]:
        """
        Get GPU statistics for all available GPUs.
        
        Returns:
            Dictionary mapping GPU ID to GPUStats if GPUs are available, None otherwise
        """
        if not self.has_gpu:
            return None
            
        gpu_stats = {}
        for gpu in GPUtil.getGPUs():
            stats = GPUStats(
                memory_used=gpu.memoryUsed,
                memory_total=gpu.memoryTotal,
                utilization=gpu.load * 100,
                temperature=gpu.temperature
            )
            gpu_stats[gpu.id] = stats
            
        return gpu_stats
    
    def print_gpu_stats(self, message: str = ""):
        """
        Print GPU statistics with an optional message.
        
        Args:
            message: Optional message to print before the stats
        """
        if message:
            print(f"\n=== {message} ===")
        
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print("\nGPU Statistics:")
            for gpu_id, stats in gpu_stats.items():
                print(f"\nGPU {gpu_id}:")
                print(f"  Memory Used: {stats.memory_used:.2f} MB")
                print(f"  Memory Total: {stats.memory_total:.2f} MB")
                print(f"  Utilization: {stats.utilization:.2f}%")
                print(f"  Temperature: {stats.temperature:.2f}°C")
        else:
            print("\nNo GPU available")
    
    def print_model_stats(self, model: Any):
        """
        Print model statistics.
        
        Args:
            model: The model to analyze
        """
        model_stats = self.get_model_stats(model)
        print("\nModel Information:")
        print(f"  Name: {model_stats.model_name}")
        print(f"  Size: {model_stats.model_size:.2f} MB")
        print(f"  Parameters: {model_stats.num_parameters:,}")
        print(f"  Device: {model_stats.device}")
        print(f"  Precision: {model_stats.precision}")
        if model_stats.quantization:
            print(f"  Quantization: {model_stats.quantization}")
        else:
            print("  Quantization: None")
    
    def get_model_stats(self, model: Any) -> ModelStats:
        """
        Get statistics about the model.
        
        Args:
            model: The model to analyze
            
        Returns:
            ModelStats object containing model information
        """
        if hasattr(model, 'config'):
            model_name = model.config.name_or_path
        else:
            model_name = model.__class__.__name__
            
        # Get model size
        model_size = self._get_model_size(model)
        
        # Determine quantization
        quantization = self._detect_quantization(model)
        
        # Get precision
        precision = self._get_model_precision(model)
        
        # Get device
        device = next(model.parameters()).device.type
        
        # Count parameters
        num_parameters = sum(p.numel() for p in model.parameters())
        
        return ModelStats(
            model_name=model_name,
            model_size=model_size,
            quantization=quantization,
            precision=precision,
            device=device,
            num_parameters=num_parameters
        )
    
    def _get_model_size(self, model: Any) -> float:
        """Calculate model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return round(size_mb, 2)
    
    def _detect_quantization(self, model: Any) -> Optional[str]:
        """Detect if and how the model is quantized."""
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            return str(model.config.quantization_config)
        
        # Check for common quantization methods
        for module in model.modules():
            if 'Int8' in str(module.__class__):
                return '8-bit quantization'
            elif 'Int4' in str(module.__class__):
                return '4-bit quantization'
            elif 'QInt' in str(module.__class__):
                return 'Dynamic quantization'
                
        return None
    
    def _get_model_precision(self, model: Any) -> str:
        """Determine model precision (fp16, fp32, bf16, etc.)."""
        dtype = next(model.parameters()).dtype
        if dtype == torch.float16:
            return 'fp16'
        elif dtype == torch.bfloat16:
            return 'bf16'
        elif dtype == torch.float32:
            return 'fp32'
        elif dtype == torch.float64:
            return 'fp64'
        else:
            return str(dtype)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary containing memory usage information in MB
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'shared': memory_info.shared / 1024 / 1024,  # Shared Memory
            'data': memory_info.data / 1024 / 1024,  # Data Memory
            'system_total': psutil.virtual_memory().total / 1024 / 1024,
            'system_available': psutil.virtual_memory().available / 1024 / 1024
        }
        
    def print_system_info(self, model: Any = None):
        """Print comprehensive system information."""
        print("\n=== System Information ===")
        
        # GPU Information
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print("\nGPU Statistics:")
            for gpu_id, stats in gpu_stats.items():
                print(f"\nGPU {gpu_id}:")
                print(f"  Memory Used: {stats.memory_used:.2f} MB")
                print(f"  Memory Total: {stats.memory_total:.2f} MB")
                print(f"  Utilization: {stats.utilization:.2f}%")
                print(f"  Temperature: {stats.temperature:.2f}°C")
        else:
            print("\nNo GPU available")
            
        # Memory Usage
        mem_usage = self.get_memory_usage()
        print("\nMemory Usage:")
        print(f"  RSS: {mem_usage['rss']:.2f} MB")
        print(f"  VMS: {mem_usage['vms']:.2f} MB")
        print(f"  System Total: {mem_usage['system_total']:.2f} MB")
        print(f"  System Available: {mem_usage['system_available']:.2f} MB")
        
        # Model Information
        if model is not None:
            model_stats = self.get_model_stats(model)
            print("\nModel Information:")
            print(f"  Name: {model_stats.model_name}")
            print(f"  Size: {model_stats.model_size:.2f} MB")
            print(f"  Parameters: {model_stats.num_parameters:,}")
            print(f"  Device: {model_stats.device}")
            print(f"  Precision: {model_stats.precision}")
            if model_stats.quantization:
                print(f"  Quantization: {model_stats.quantization}")
            else:
                print("  Quantization: None") 