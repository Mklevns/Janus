"""
Checkpoint Manager for Janus
============================

Handles saving and loading of training checkpoints for recovery.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import numpy as np
from datetime import datetime
import logging


class CheckpointManager:
    """Manages checkpoint saving and loading for training recovery."""

    def __init__(self, checkpoint_dir: Union[str, Path]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # Metadata file to track checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        iteration: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint with metadata.

        Args:
            state: Dictionary containing all state to save
            iteration: Current iteration number
            metrics: Optional metrics to save with checkpoint
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint_iter_{iteration}_{timestamp[:19].replace(':', '-')}.pkl"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint_data = {
            'state': state,
            'iteration': iteration,
            'timestamp': timestamp,
            'metrics': metrics or {}
        }

        # Handle PyTorch tensors
        checkpoint_data = self._prepare_for_pickle(checkpoint_data)

        # Save checkpoint
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Update metadata
            self._update_metadata(checkpoint_name, iteration, metrics, is_best)

            # Save best checkpoint separately
            if is_best:
                best_path = self.checkpoint_dir / "checkpoint_best.pkl"
                with open(best_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Clean up old checkpoints
            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        if not self.metadata.get('checkpoints'):
            self.logger.warning("No checkpoints found")
            return None

        # Sort by iteration number
        latest_checkpoint = max(
            self.metadata['checkpoints'],
            key=lambda x: x['iteration']
        )

        return self.load_checkpoint(latest_checkpoint['filename'])

    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pkl"
        if best_path.exists():
            return self.load_checkpoint("checkpoint_best.pkl")

        # Fallback to metadata
        best_checkpoints = [
            cp for cp in self.metadata.get('checkpoints', [])
            if cp.get('is_best', False)
        ]

        if best_checkpoints:
            return self.load_checkpoint(best_checkpoints[-1]['filename'])

        return None

    def load_checkpoint(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a specific checkpoint by filename."""
        checkpoint_path = self.checkpoint_dir / filename

        if not checkpoint_path.exists():
            self.logger.error(f"Checkpoint {filename} not found")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            # Restore PyTorch tensors if needed
            checkpoint_data = self._restore_from_pickle(checkpoint_data)

            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        return self.metadata.get('checkpoints', [])

    def _prepare_for_pickle(self, obj: Any) -> Any:
        """Prepare object for pickling (handle PyTorch tensors)."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_pickle(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_pickle(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._prepare_for_pickle(item) for item in obj)
        elif torch.is_tensor(obj):
            return {'_tensor': True, 'data': obj.cpu().numpy(), 'dtype': str(obj.dtype)}
        else:
            return obj

    def _restore_from_pickle(self, obj: Any) -> Any:
        """Restore object after unpickling (handle PyTorch tensors)."""
        if isinstance(obj, dict):
            if obj.get('_tensor'):
                # Restore tensor
                data = obj['data']
                dtype = getattr(torch, obj['dtype'].split('.')[-1])
                return torch.tensor(data, dtype=dtype)
            else:
                return {k: self._restore_from_pickle(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._restore_from_pickle(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._restore_from_pickle(item) for item in obj)
        else:
            return obj

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'checkpoints': [], 'best_iteration': None}

    def _update_metadata(
        self,
        filename: str,
        iteration: int,
        metrics: Optional[Dict[str, float]],
        is_best: bool
    ) -> None:
        """Update checkpoint metadata."""
        checkpoint_info = {
            'filename': filename,
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'is_best': is_best
        }

        self.metadata['checkpoints'].append(checkpoint_info)

        if is_best:
            self.metadata['best_iteration'] = iteration

        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _cleanup_old_checkpoints(self, keep_n: int = 5) -> None:
        """Clean up old checkpoints, keeping only the most recent ones."""
        checkpoints = self.metadata.get('checkpoints', [])

        if len(checkpoints) <= keep_n:
            return

        # Sort by iteration
        checkpoints.sort(key=lambda x: x['iteration'])

        # Keep best and most recent
        to_remove = []
        for cp in checkpoints[:-keep_n]:
            if not cp.get('is_best'):
                to_remove.append(cp)

        # Remove files and update metadata
        for cp in to_remove:
            checkpoint_path = self.checkpoint_dir / cp['filename']
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            self.metadata['checkpoints'].remove(cp)

        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
