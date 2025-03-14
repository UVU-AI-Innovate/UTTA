from typing import List, Dict, Any, Optional
import json
from pathlib import Path
from ..metrics.automated_metrics import AutomatedMetrics
from datetime import datetime
import yaml

class BenchmarkSuite:
    """Class for running benchmark tests on the chatbot."""
    
    def __init__(self, benchmark_dir: str = "benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = AutomatedMetrics()
        
    def load_benchmark_set(self, benchmark_file: str) -> List[Dict[str, str]]:
        """
        Load a benchmark test set from a YAML file.
        
        Args:
            benchmark_file: Path to the YAML file containing benchmark tests
            
        Returns:
            List of benchmark test cases
        """
        benchmark_path = self.benchmark_dir / benchmark_file
        
        if not benchmark_path.exists():
            raise FileNotFoundError(f"Benchmark file {benchmark_file} not found")
            
        with open(benchmark_path, 'r') as f:
            benchmark_data = yaml.safe_load(f)
            
        return benchmark_data['test_cases']
    
    def create_benchmark_set(self, 
                           test_cases: List[Dict[str, str]], 
                           name: str,
                           description: str = "") -> None:
        """
        Create a new benchmark test set.
        
        Args:
            test_cases: List of test cases with input and expected output
            name: Name of the benchmark set
            description: Optional description of the benchmark set
        """
        benchmark_data = {
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "test_cases": test_cases
        }
        
        output_file = self.benchmark_dir / f"{name}.yaml"
        with open(output_file, 'w') as f:
            yaml.dump(benchmark_data, f, sort_keys=False)
            
    def run_benchmark(self, 
                     chatbot,
                     benchmark_name: str,
                     output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run benchmark tests on a chatbot instance.
        
        Args:
            chatbot: Instance of the chatbot to test
            benchmark_name: Name of the benchmark set to run
            output_file: Optional file to save results
            
        Returns:
            Dictionary containing benchmark results
        """
        test_cases = self.load_benchmark_set(f"{benchmark_name}.yaml")
        results = []
        
        for test_case in test_cases:
            # Generate response from chatbot
            response = chatbot.generate_response(test_case['input'])
            
            # Calculate metrics
            metrics = self.metrics.evaluate_response(
                response,
                test_case['expected_output']
            )
            
            results.append({
                "input": test_case['input'],
                "expected": test_case['expected_output'],
                "generated": response,
                "metrics": metrics
            })
            
        # Calculate aggregate scores
        aggregate_scores = {
            metric: sum(r['metrics'][metric] for r in results) / len(results)
            for metric in results[0]['metrics'].keys()
        }
        
        benchmark_results = {
            "benchmark_name": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "aggregate_scores": aggregate_scores,
            "detailed_results": results
        }
        
        if output_file:
            output_path = self.benchmark_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
                
        return benchmark_results
        
    def compare_benchmarks(self, 
                          result_files: List[str]) -> Dict[str, Any]:
        """
        Compare results from multiple benchmark runs.
        
        Args:
            result_files: List of benchmark result files to compare
            
        Returns:
            Dictionary containing comparison metrics
        """
        all_results = []
        
        for file in result_files:
            file_path = self.benchmark_dir / file
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)
                
        # Compare aggregate scores
        comparison = {
            "benchmarks": [r["benchmark_name"] for r in all_results],
            "timestamps": [r["timestamp"] for r in all_results],
            "metric_comparisons": {}
        }
        
        # For each metric, compare scores across benchmark runs
        metrics = all_results[0]["aggregate_scores"].keys()
        for metric in metrics:
            comparison["metric_comparisons"][metric] = {
                "scores": [r["aggregate_scores"][metric] for r in all_results],
                "diff": max(r["aggregate_scores"][metric] for r in all_results) - 
                       min(r["aggregate_scores"][metric] for r in all_results)
            }
            
        return comparison 