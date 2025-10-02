#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time Estimator for FAIRChem Training
====================================

This module provides a time estimation callback that can be used with FAIRChem's
TrainEvalRunner to provide real-time estimates of training completion time.

Usage:
    from src_v2.time_estimator import TimeEstimationCallback
    
    # Add to your training callbacks
    callbacks = [
        TimeEstimationCallback(
            update_interval=100,  # Update every 100 steps
            enable_detailed_logging=True
        ),
        # ... other callbacks
    ]
    
    runner = TrainEvalRunner(
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        train_eval_unit=unit,
        callbacks=callbacks,
        max_epochs=100,
        max_steps=10000
    )
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
import torch

class TimeEstimationCallback(Callback):
    """
    A callback that provides real-time time estimation for training completion.
    
    This callback tracks training progress and provides estimates for:
    - Time per step/epoch
    - Estimated completion time
    - Progress percentage
    - Remaining time
    """
    
    def __init__(
        self,
        update_interval: int = 100,
        enable_detailed_logging: bool = True,
        log_format: str = "detailed",
        max_steps: int = None,
        max_epochs: int = None,
        train_dataloader: torch.utils.data.dataloader = None,
        eval_dataloader: torch.utils.data.dataloader = None,
        evaluate_every_n_steps: int = None
    ):
        """
        Initialize the time estimation callback.
        
        Args:
            update_interval: How often to update time estimates (in steps)
            enable_detailed_logging: Whether to log detailed time information
            log_format: Format for logging ('detailed', 'compact', 'minimal')
            max_steps: Maximum number of steps (optional)
            max_epochs: Maximum number of epochs (optional)
            train_dataloader: Training dataloader for step calculation
            eval_dataloader: Evaluation dataloader for step calculation
            evaluate_every_n_steps: How often to evaluate (for total step calculation)
        """
        self.update_interval = update_interval
        self.enable_detailed_logging = enable_detailed_logging
        self.log_format = log_format
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.evaluate_every_n_steps = evaluate_every_n_steps
        # Calculate total steps if dataloaders are provided
        if train_dataloader is not None and eval_dataloader is not None and evaluate_every_n_steps is not None:
            self.train_dataloader_length = len(train_dataloader)
            self.eval_dataloader_length = len(eval_dataloader)
            self.train_steps_per_epoch = self.train_dataloader_length
            
            if self.max_epochs:
                self.total_train_steps = self.train_steps_per_epoch * self.max_epochs
                self.total_eval_steps = (self.total_train_steps // self.evaluate_every_n_steps) * self.eval_dataloader_length
                self.calculated_total_steps = self.total_train_steps + self.total_eval_steps
            else:
                self.calculated_total_steps = None
        else:
            self.calculated_total_steps = None
        
        # Time tracking variables
        self.start_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.step_times: list = []
        self.epoch_times: list = []
        
        # Progress tracking - use calculated values or provided max_steps
        self.total_steps = self.calculated_total_steps if self.calculated_total_steps is not None else self.max_steps
        self.total_epochs = self.max_epochs
        self.current_step: int = 0
        self.current_epoch: int = 0
        
        # Performance metrics
        self.avg_time_per_step: float = 0.0
        self.avg_time_per_epoch: float = 0.0
        
        logging.info(f"TimeEstimationCallback initialized with update_interval={update_interval}")
        if self.total_steps:
            logging.info(f"Total steps calculated: {self.total_steps}")
        if self.total_epochs:
            logging.info(f"Total epochs planned: {self.total_epochs}")
    
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Don't override calculated values - they're already set in __init__
        # Only update if we don't have them yet
        if self.total_steps is None:
            self.total_steps = self.max_steps
        if self.total_epochs is None:
            self.total_epochs = self.max_epochs
        
        logging.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.total_steps:
            logging.info(f"Total steps planned: {self.total_steps}")
        if self.total_epochs:
            logging.info(f"Total epochs planned: {self.total_epochs}")
    
    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        """Called at the start of each training step."""
        step_start_time = time.time()
        
        # Update current step
        self.current_step = getattr(unit.train_progress, 'num_steps_completed', 0)
        
        # Store step start time for duration calculation
        if not hasattr(self, '_step_start_times'):
            self._step_start_times = {}
        self._step_start_times[self.current_step] = step_start_time
        
        # Update time estimates periodically
        if self.current_step % self.update_interval == 0:
            self._update_time_estimates()
    
    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        """Called at the end of each training step."""
        step_end_time = time.time()
        
        # Calculate step duration
        if hasattr(self, '_step_start_times') and self.current_step in self._step_start_times:
            step_duration = step_end_time - self._step_start_times[self.current_step]
            self.step_times.append(step_duration)
            
            # Keep only recent step times for rolling average
            if len(self.step_times) > 1000:
                self.step_times = self.step_times[-1000:]
    
    def on_train_epoch_start(self, state: State, unit: TTrainUnit) -> None:
        """Called at the start of each training epoch."""
        epoch_start_time = time.time()
        
        # Update current epoch
        self.current_epoch = getattr(unit.train_progress, 'num_epochs_completed', 0)
        
        # Store epoch start time
        if not hasattr(self, '_epoch_start_times'):
            self._epoch_start_times = {}
        self._epoch_start_times[self.current_epoch] = epoch_start_time
        
        logging.info(f"Starting epoch {self.current_epoch}")
    
    def on_train_epoch_end(self, state: State, unit: TTrainUnit) -> None:
        """Called at the end of each training epoch."""
        epoch_end_time = time.time()
        
        # Calculate epoch duration
        if hasattr(self, '_epoch_start_times') and self.current_epoch in self._epoch_start_times:
            epoch_duration = epoch_end_time - self._epoch_start_times[self.current_epoch]
            self.epoch_times.append(epoch_duration)
            
            # Keep only recent epoch times for rolling average
            if len(self.epoch_times) > 100:
                self.epoch_times = self.epoch_times[-100:]
            
            # Log epoch completion with time estimate
            self._log_epoch_completion(epoch_duration)
    
    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        """Called when training ends."""
        if self.start_time is None:
            return
            
        total_training_time = time.time() - self.start_time
        
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED")
        logging.info("=" * 60)
        logging.info(f"Total training time: {timedelta(seconds=int(total_training_time))}")
        logging.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            logging.info(f"Average time per step: {avg_step_time:.3f}s")
        
        if self.epoch_times:
            avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
            logging.info(f"Average time per epoch: {avg_epoch_time:.3f}s")
    
    def _update_time_estimates(self) -> None:
        """Update and log time estimates."""
        if self.start_time is None:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate averages
        if self.step_times:
            self.avg_time_per_step = sum(self.step_times) / len(self.step_times)
        
        if self.epoch_times:
            self.avg_time_per_epoch = sum(self.epoch_times) / len(self.epoch_times)
        
        # Estimate remaining time
        estimated_remaining = self._estimate_remaining_time() # this is in seconds

        # Calculate completion time
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
        
        # Log estimates
        self._log_time_estimates(
            elapsed_time, 
            estimated_remaining, 
            completion_time
        )
        
        self.last_update_time = current_time
    
    def _estimate_remaining_time(self) -> float:
        """Estimate remaining training time based on epochs."""
        if self.total_steps and self.avg_time_per_step > 0:
            # Fallback: estimate based on steps

            remaining_steps = self.total_steps - self.current_step
            return remaining_steps * self.avg_time_per_step
        else:
            # Fallback: estimate based on progress ratio using epochs
            if self.start_time and self.last_update_time:
                elapsed = self.last_update_time - self.start_time
                if self.current_epoch > 0:
                    # Rough estimate: assume linear progress based on epochs
                    progress_ratio = self.current_epoch / max(self.total_epochs or 1, 1)
                    if progress_ratio > 0:
                        total_estimated = elapsed / progress_ratio
                        return total_estimated - elapsed
            return 0.0
    
    def _log_time_estimates(
        self, 
        elapsed_time: float, 
        estimated_remaining: float, 
        completion_time: datetime
    ) -> None:
        """Log time estimates in the specified format."""
        if self.log_format == "minimal":
            self._log_minimal(elapsed_time, estimated_remaining, completion_time)
        elif self.log_format == "compact":
            self._log_compact(elapsed_time, estimated_remaining, completion_time)
        else:  # detailed
            self._log_detailed(elapsed_time, estimated_remaining, completion_time)
    
    def _log_minimal(
        self, 
        elapsed_time: float, 
        estimated_remaining: float, 
        completion_time: datetime
    ) -> None:
        """Log minimal time information."""
        logging.info(
            f"Step {self.current_step} | "
            f"ETA: {completion_time.strftime('%H:%M:%S')} | "
            f"Remaining: {timedelta(seconds=int(estimated_remaining))}"
        )
    
    def _log_compact(
        self, 
        elapsed_time: float, 
        estimated_remaining: float, 
        completion_time: datetime
    ) -> None:
        """Log compact time information."""
        progress = 0.0
        if self.total_epochs:
            progress = (self.current_epoch / self.total_epochs) * 100
        elif self.total_steps:
            progress = (self.current_step / self.total_steps) * 100
        
        if self.total_epochs:
            logging.info(
                f"Epoch {self.current_epoch}/{self.total_epochs} "
                f"({progress:.1f}%) | "
                f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
                f"ETA: {completion_time.strftime('%H:%M:%S')} | "
                f"Remaining: {timedelta(seconds=int(estimated_remaining))}"
            )
        else:
            logging.info(
                f"Step {self.current_step}/{self.total_steps or '?'} "
                f"({progress:.1f}%) | "
                f"Elapsed: {timedelta(seconds=int(elapsed_time))} | "
                f"ETA: {completion_time.strftime('%H:%M:%S')} | "
                f"Remaining: {timedelta(seconds=int(estimated_remaining))}"
            )
    
    def _log_detailed(
        self, 
        elapsed_time: float, 
        estimated_remaining: float, 
        completion_time: datetime
    ) -> None:
        """Log detailed time information."""
        progress = 0.0
        if self.total_epochs:
            progress = (self.current_epoch / self.total_epochs) * 100
        elif self.total_steps:
            progress = (self.current_step / self.total_steps) * 100
        
        logging.info("=" * 50)
        logging.info("TIME ESTIMATION UPDATE")
        logging.info("=" * 50)
        logging.info(f"Current Epoch: {self.current_epoch}")
        if self.total_epochs:
            logging.info(f"Total Epochs: {self.total_epochs}")
            logging.info(f"Progress: {progress:.1f}%")
        
        if self.total_steps:
            logging.info(f"Current Step: {self.current_step}")
            logging.info(f"Total Steps: {self.total_steps}")
        
        logging.info(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}")
        logging.info(f"Estimated Remaining: {timedelta(seconds=int(estimated_remaining))}")
        logging.info(f"Estimated Completion: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.avg_time_per_epoch > 0:
            logging.info(f"Average Time per Epoch: {self.avg_time_per_epoch:.3f}s")
        
        if self.avg_time_per_step > 0:
            logging.info(f"Average Time per Step: {self.avg_time_per_step:.3f}s")
        
        logging.info("=" * 50)
    
    def _log_epoch_completion(self, epoch_duration: float) -> None:
        """Log epoch completion with time information."""
        logging.info(
            f"Epoch {self.current_epoch} completed in "
            f"{timedelta(seconds=int(epoch_duration))}"
        )
        
        # Update time estimates after epoch completion
        if self.enable_detailed_logging:
            self._update_time_estimates()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current time estimation statistics."""
        if self.start_time is None:
            return {}
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        estimated_remaining = self._estimate_remaining_time()
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
        
        progress = 0.0
        if self.total_epochs:
            progress = (self.current_epoch / self.total_epochs) * 100
        elif self.total_steps:
            progress = (self.current_step / self.total_steps) * 100
        
        return {
            "current_step": self.current_step,
            "current_epoch": self.current_epoch,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "progress_percent": progress,
            "elapsed_time": elapsed_time,
            "elapsed_time_formatted": str(timedelta(seconds=int(elapsed_time))),
            "estimated_remaining": estimated_remaining,
            "estimated_remaining_formatted": str(timedelta(seconds=int(estimated_remaining))),
            "estimated_completion": completion_time,
            "estimated_completion_formatted": completion_time.strftime('%Y-%m-%d %H:%M:%S'),
            "avg_time_per_step": self.avg_time_per_step,
            "avg_time_per_epoch": self.avg_time_per_epoch,
            "start_time": datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S') if self.start_time else None
        }
