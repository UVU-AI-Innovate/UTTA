"""Automated diagnostics for the teaching assistant application."""

import os
import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: bool
    message: str
    error: Exception = None
    fix_applied: bool = False
    fix_message: str = ""

class SystemDiagnostics:
    """System-wide diagnostics for the teaching assistant."""
    
    def __init__(self):
        """Initialize diagnostics."""
        self.results: List[DiagnosticResult] = []
        self.fixes_available = {
            "openai_key": self._fix_openai_key,
            "dspy_install": self._fix_dspy_install,
            "spacy_model": self._fix_spacy_model,
            "torch_config": self._fix_torch_config
        }

    @contextmanager
    def _capture_logs(self):
        """Capture logs during a diagnostic check."""
        log_capture = []
        
        class LogHandler(logging.Handler):
            def emit(self, record):
                log_capture.append(record)
        
        handler = LogHandler()
        logger.addHandler(handler)
        try:
            yield log_capture
        finally:
            logger.removeHandler(handler)

    def check_environment(self) -> DiagnosticResult:
        """Check environment variables."""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                return DiagnosticResult(
                    name="environment",
                    status=False,
                    message="OpenAI API key not found in environment"
                )
            return DiagnosticResult(
                name="environment",
                status=True,
                message="Environment variables configured correctly"
            )
        except Exception as e:
            return DiagnosticResult(
                name="environment",
                status=False,
                message=f"Error checking environment: {str(e)}",
                error=e
            )

    def check_dependencies(self) -> DiagnosticResult:
        """Check required package dependencies."""
        required_packages = {
            "streamlit": "1.32.0",
            "dspy-ai": "2.0.4",
            "torch": "2.0.0",
            "spacy": "3.7.0",
            "sentence-transformers": "2.2.2"
        }
        
        missing = []
        outdated = []
        
        for package, version in required_packages.items():
            try:
                imported = importlib.import_module(package.replace("-", "_"))
                if hasattr(imported, "__version__"):
                    current = imported.__version__
                    if current != version:
                        outdated.append(f"{package} (have {current}, need {version})")
            except ImportError:
                missing.append(package)
        
        if missing or outdated:
            msg = []
            if missing:
                msg.append(f"Missing packages: {', '.join(missing)}")
            if outdated:
                msg.append(f"Outdated packages: {', '.join(outdated)}")
            return DiagnosticResult(
                name="dependencies",
                status=False,
                message=" | ".join(msg)
            )
        
        return DiagnosticResult(
            name="dependencies",
            status=True,
            message="All dependencies installed and up to date"
        )

    def check_spacy_model(self) -> DiagnosticResult:
        """Check spaCy model installation."""
        try:
            import spacy
            if not spacy.util.is_package("en_core_web_sm"):
                return DiagnosticResult(
                    name="spacy_model",
                    status=False,
                    message="spaCy English model not installed"
                )
            nlp = spacy.load("en_core_web_sm")
            return DiagnosticResult(
                name="spacy_model",
                status=True,
                message="spaCy model loaded successfully"
            )
        except Exception as e:
            return DiagnosticResult(
                name="spacy_model",
                status=False,
                message=f"Error loading spaCy model: {str(e)}",
                error=e
            )

    def check_torch_config(self) -> DiagnosticResult:
        """Check PyTorch configuration."""
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type == 'cuda':
                torch.cuda.init()
                gpu_name = torch.cuda.get_device_name(0)
                msg = f"PyTorch configured for GPU: {gpu_name}"
            else:
                msg = "PyTorch configured for CPU"
            
            return DiagnosticResult(
                name="torch_config",
                status=True,
                message=msg
            )
        except Exception as e:
            return DiagnosticResult(
                name="torch_config",
                status=False,
                message=f"Error configuring PyTorch: {str(e)}",
                error=e
            )

    def check_component_initialization(self) -> List[DiagnosticResult]:
        """Check initialization of main components."""
        results = []
        
        # Check LLM Interface
        with self._capture_logs() as logs:
            try:
                from llm.dspy.handler import EnhancedDSPyLLMInterface
                llm = EnhancedDSPyLLMInterface()
                results.append(DiagnosticResult(
                    name="llm_interface",
                    status=True,
                    message="LLM interface initialized successfully"
                ))
            except Exception as e:
                results.append(DiagnosticResult(
                    name="llm_interface",
                    status=False,
                    message=f"Failed to initialize LLM interface: {str(e)}",
                    error=e
                ))
        
        # Check Document Indexer
        with self._capture_logs() as logs:
            try:
                from knowledge_base.document_indexer import DocumentIndexer
                indexer = DocumentIndexer()
                results.append(DiagnosticResult(
                    name="document_indexer",
                    status=True,
                    message="Document indexer initialized successfully"
                ))
            except Exception as e:
                results.append(DiagnosticResult(
                    name="document_indexer",
                    status=False,
                    message=f"Failed to initialize document indexer: {str(e)}",
                    error=e
                ))
        
        # Check Metrics Evaluator
        with self._capture_logs() as logs:
            try:
                from metrics.automated_metrics import AutomatedMetricsEvaluator
                evaluator = AutomatedMetricsEvaluator()
                results.append(DiagnosticResult(
                    name="metrics_evaluator",
                    status=True,
                    message="Metrics evaluator initialized successfully"
                ))
            except Exception as e:
                results.append(DiagnosticResult(
                    name="metrics_evaluator",
                    status=False,
                    message=f"Failed to initialize metrics evaluator: {str(e)}",
                    error=e
                ))
        
        return results

    def _fix_openai_key(self) -> bool:
        """Fix OpenAI API key issue."""
        try:
            key = input("Enter your OpenAI API key: ").strip()
            if key:
                with open(".env", "a") as f:
                    f.write(f"\nOPENAI_API_KEY={key}\n")
                os.environ["OPENAI_API_KEY"] = key
                return True
        except Exception as e:
            logger.error(f"Failed to fix OpenAI key: {e}")
        return False

    def _fix_dspy_install(self) -> bool:
        """Fix DSPy installation."""
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "dspy-ai==2.0.4"])
            return True
        except Exception as e:
            logger.error(f"Failed to install DSPy: {e}")
        return False

    def _fix_spacy_model(self) -> bool:
        """Fix spaCy model installation."""
        try:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            return True
        except Exception as e:
            logger.error(f"Failed to download spaCy model: {e}")
        return False

    def _fix_torch_config(self) -> bool:
        """Fix PyTorch configuration."""
        try:
            import torch
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                torch.cuda.init()
            return True
        except Exception as e:
            logger.error(f"Failed to configure PyTorch: {e}")
        return False

    def run_diagnostics(self, auto_fix: bool = False) -> List[DiagnosticResult]:
        """Run all diagnostics and optionally attempt to fix issues.
        
        Args:
            auto_fix: Whether to attempt automatic fixes for issues
            
        Returns:
            List of diagnostic results
        """
        # Clear previous results
        self.results = []
        
        # Run environment checks
        env_result = self.check_environment()
        self.results.append(env_result)
        if not env_result.status and auto_fix:
            if self._fix_openai_key():
                env_result.fix_applied = True
                env_result.fix_message = "Added OpenAI API key to environment"
        
        # Run dependency checks
        dep_result = self.check_dependencies()
        self.results.append(dep_result)
        if not dep_result.status and auto_fix:
            if "dspy-ai" in dep_result.message and self._fix_dspy_install():
                dep_result.fix_applied = True
                dep_result.fix_message = "Installed/updated DSPy package"
        
        # Run spaCy model check
        spacy_result = self.check_spacy_model()
        self.results.append(spacy_result)
        if not spacy_result.status and auto_fix:
            if self._fix_spacy_model():
                spacy_result.fix_applied = True
                spacy_result.fix_message = "Downloaded spaCy English model"
        
        # Run PyTorch config check
        torch_result = self.check_torch_config()
        self.results.append(torch_result)
        if not torch_result.status and auto_fix:
            if self._fix_torch_config():
                torch_result.fix_applied = True
                torch_result.fix_message = "Configured PyTorch settings"
        
        # Run component checks
        self.results.extend(self.check_component_initialization())
        
        return self.results

    def print_report(self):
        """Print a formatted diagnostic report."""
        print("\n=== Teaching Assistant Diagnostic Report ===\n")
        
        for result in self.results:
            status = "✅" if result.status else "❌"
            print(f"{status} {result.name}:")
            print(f"   {result.message}")
            
            if result.error:
                print(f"   Error details: {str(result.error)}")
                print(f"   Traceback: {traceback.format_tb(result.error.__traceback__)[-1]}")
            
            if result.fix_applied:
                print(f"   🔧 Fix applied: {result.fix_message}")
            
            print()
        
        # Print summary
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status)
        print(f"\nSummary: {passed}/{total} checks passed")
        
        if passed < total:
            print("\nRecommended actions:")
            for result in self.results:
                if not result.status:
                    if result.name in self.fixes_available:
                        print(f"- {result.name}: Automatic fix available")
                    else:
                        print(f"- {result.name}: Manual intervention required")

def main():
    """Run diagnostics from command line."""
    import argparse
    parser = argparse.ArgumentParser(description="Run teaching assistant diagnostics")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    args = parser.parse_args()
    
    diagnostics = SystemDiagnostics()
    diagnostics.run_diagnostics(auto_fix=args.fix)
    diagnostics.print_report()

if __name__ == "__main__":
    main() 