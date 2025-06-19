# janus/experiments/base.py
"""
Enhanced base experiment class with registry integration.
"""

import abc
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle

from janus.config.models import ExperimentConfig, ExperimentResult
from janus.utils.logging import get_logger
from janus.utils.metrics import MetricsTracker

logger = get_logger(__name__)


@dataclass
class ExperimentContext:
    """Context information for experiment execution."""
    run_id: int = 0
    total_runs: int = 1
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    wandb_run: Optional[Any] = None
    metrics_tracker: Optional[MetricsTracker] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseExperiment(abc.ABC):
    """
    Enhanced base class for all Janus experiments.
    
    Provides:
    - Automatic metric tracking
    - Checkpointing support
    - Result serialization
    - Integration with experiment registry
    """
    
    def __init__(self, 
                 config: ExperimentConfig,
                 context: Optional[ExperimentContext] = None,
                 **kwargs):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
            context: Execution context (created if not provided)
            **kwargs: Additional experiment-specific arguments
        """
        self.config = config
        self.context = context or ExperimentContext()
        self.kwargs = kwargs
        
        self._start_time = None
        self._end_time = None
        self.results = None
        self.metrics = MetricsTracker()
        
        # Setup logging
        self.logger = get_logger(self.__class__.__name__)
        
        # Get registry metadata if available
        if hasattr(self.__class__, '_registry_metadata'):
            self.metadata = self.__class__._registry_metadata
        else:
            self.metadata = None
            
    @abc.abstractmethod
    def setup(self):
        """
        Setup experiment environment and resources.
        
        This method should:
        - Initialize environments
        - Load models
        - Prepare data
        - Set random seeds
        """
        pass
        
    @abc.abstractmethod  
    def run(self, run_id: int = 0) -> ExperimentResult:
        """
        Execute the main experiment logic.
        
        Args:
            run_id: Current run identifier
            
        Returns:
            ExperimentResult containing discoveries and metrics
        """
        pass
        
    @abc.abstractmethod
    def teardown(self):
        """
        Clean up resources after experiment.
        
        This method should:
        - Close environments
        - Free GPU memory
        - Save final results
        - Close logging handlers
        """
        pass
        
    def execute(self) -> ExperimentResult:
        """
        Execute the complete experiment workflow.
        
        Returns:
            Final experiment result
        """
        self.logger.info(f"Starting experiment: {self.config.name}")
        self._start_time = time.time()
        
        try:
            # Setup phase
            self.logger.info("Setting up experiment...")
            self.setup()
            self.metrics.log("setup_complete", True)
            
            # Run phase
            self.logger.info(f"Running experiment (run {self.context.run_id + 1}/{self.context.total_runs})...")
            self.results = self.run(self.context.run_id)
            self.metrics.log("run_complete", True)
            
            # Record timing
            self._end_time = time.time()
            self.results.wall_time_seconds = self._end_time - self._start_time
            
            # Log final metrics
            self._log_final_metrics()
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            self.metrics.log("error", str(e))
            
            # Create error result
            self.results = ExperimentResult(
                config=self.config,
                run_id=self.context.run_id,
                discovered_law="ERROR",
                error_message=str(e)
            )
            raise
            
        finally:
            # Teardown phase
            self.logger.info("Tearing down experiment...")
            try:
                self.teardown()
            except Exception as e:
                self.logger.error(f"Teardown failed: {e}", exc_info=True)
                
            # Save results
            if self.results and self.context.output_dir:
                self._save_results()
                
        self.logger.info(f"Experiment complete in {self.results.wall_time_seconds:.2f}s")
        return self.results
        
    def checkpoint(self, checkpoint_name: Optional[str] = None):
        """Save experiment checkpoint."""
        if not self.context.checkpoint_dir:
            return
            
        checkpoint_name = checkpoint_name or f"checkpoint_run{self.context.run_id}"
        checkpoint_path = self.context.checkpoint_dir / f"{checkpoint_name}.pkl"
        
        checkpoint_data = {
            'config': self.config,
            'context': self.context,
            'metrics': self.metrics.get_all(),
            'timestamp': time.time()
        }
        
        # Allow experiments to add custom checkpoint data
        if hasattr(self, 'get_checkpoint_data'):
            checkpoint_data['custom'] = self.get_checkpoint_data()
            
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: Path):
        """Load experiment checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        # Restore metrics
        for key, value in checkpoint_data['metrics'].items():
            self.metrics.log(key, value)
            
        # Allow experiments to restore custom data
        if hasattr(self, 'restore_checkpoint_data') and 'custom' in checkpoint_data:
            self.restore_checkpoint_data(checkpoint_data['custom'])
            
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
        
    def _log_final_metrics(self):
        """Log final experiment metrics."""
        if self.results:
            self.metrics.log("discovered_law", self.results.discovered_law)
            self.metrics.log("accuracy", self.results.symbolic_accuracy)
            self.metrics.log("complexity", self.results.law_complexity)
            self.metrics.log("wall_time", self.results.wall_time_seconds)
            
    def _save_results(self):
        """Save experiment results."""
        # Save as JSON
        results_json = self.context.output_dir / f"results_run{self.context.run_id}.json"
        with open(results_json, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
            
        # Save as pickle for full object
        results_pkl = self.context.output_dir / f"results_run{self.context.run_id}.pkl"
        with open(results_pkl, 'wb') as f:
            pickle.dump(self.results, f)
            
        # Save metrics
        metrics_json = self.context.output_dir / f"metrics_run{self.context.run_id}.json"
        with open(metrics_json, 'w') as f:
            json.dump(self.metrics.get_all(), f, indent=2)
            
        self.logger.info(f"Saved results to {self.context.output_dir}")
