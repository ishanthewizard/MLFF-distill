#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Time Estimator with FAIRChem
============================================

This script demonstrates how to integrate the TimeEstimationCallback
with FAIRChem's TrainEvalRunner for time estimation during training.
"""

from src_v2.time_estimator import TimeEstimationCallback
from fairchem.core.components.train.train_runner import TrainEvalRunner, TrainCheckpointCallback

def create_training_runner_with_time_estimation(
    train_dataloader,
    eval_dataloader,
    train_eval_unit,
    max_epochs=100,
    max_steps=10000,
    checkpoint_every_n_steps=1000,
    time_update_interval=100,
    log_format="detailed"
):
    """
    Create a FAIRChem training runner with time estimation capabilities.
    
    Args:
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        train_eval_unit: Training/evaluation unit
        max_epochs: Maximum number of epochs
        max_steps: Maximum number of steps
        checkpoint_every_n_steps: How often to save checkpoints
        time_update_interval: How often to update time estimates
        log_format: Time estimation log format ('detailed', 'compact', 'minimal')
    
    Returns:
        Configured TrainEvalRunner with time estimation
    """
    
    # Create time estimation callback with epoch-based estimation
    time_callback = TimeEstimationCallback(
        update_interval=time_update_interval,
        enable_detailed_logging=True,
        log_format=log_format,
        max_steps=max_steps,              # Pass max_steps for fallback estimation
        max_epochs=max_epochs,            # Pass max_epochs for primary estimation
        train_dataloader=train_dataloader, # For calculating total steps
        eval_dataloader=eval_dataloader,   # For calculating total steps
        evaluate_every_n_steps=500        # How often to evaluate
    )
    
    # Create checkpoint callback
    checkpoint_callback = TrainCheckpointCallback(
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        max_saved_checkpoints=3
    )
    
    # Combine all callbacks
    callbacks = [
        time_callback,
        checkpoint_callback,
        # Add other callbacks here as needed
    ]
    
    # Create and return the training runner
    runner = TrainEvalRunner(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        train_eval_unit=train_eval_unit,
        callbacks=callbacks,
        max_epochs=max_epochs,
        max_steps=max_steps,
        evaluate_every_n_steps=500  # Evaluate every 500 steps
    )
    
    return runner, time_callback

def example_configuration():
    """
    Example configuration showing different time estimation options.
    """
    
    # Example 1: Detailed logging with frequent updates
    detailed_config = {
        "update_interval": 50,  # Update every 50 steps
        "enable_detailed_logging": True,
        "log_format": "detailed"  # Full detailed output
    }
    
    # Example 2: Compact logging with moderate updates
    compact_config = {
        "update_interval": 100,  # Update every 100 steps
        "enable_detailed_logging": True,
        "log_format": "compact"  # Condensed output
    }
    
    # Example 3: Minimal logging with infrequent updates
    minimal_config = {
        "update_interval": 500,  # Update every 500 steps
        "enable_detailed_logging": False,
        "log_format": "minimal"  # Just essential info
    }
    
    return detailed_config, compact_config, minimal_config

def get_time_estimation_stats(time_callback):
    """
    Get current time estimation statistics from the callback.
    
    Args:
        time_callback: TimeEstimationCallback instance
    
    Returns:
        Dictionary containing current statistics
    """
    return time_callback.get_current_stats()

def print_training_summary(time_callback):
    """
    Print a summary of training progress and time estimates.
    
    Args:
        time_callback: TimeEstimationCallback instance
    """
    stats = time_callback.get_current_stats()
    
    if not stats:
        print("No training statistics available yet.")
        return
    
    print("\n" + "="*60)
    print("TRAINING PROGRESS SUMMARY")
    print("="*60)
    print(f"Current Step: {stats['current_step']}")
    print(f"Current Epoch: {stats['current_epoch']}")
    
    if stats['total_steps']:
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Progress: {stats['progress_percent']:.1f}%")
    
    if stats['total_epochs']:
        print(f"Total Epochs: {stats['total_epochs']}")
    
    print(f"Elapsed Time: {stats['elapsed_time_formatted']}")
    print(f"Estimated Remaining: {stats['estimated_remaining_formatted']}")
    print(f"Estimated Completion: {stats['estimated_completion_formatted']}")
    
    if stats['avg_time_per_step'] > 0:
        print(f"Average Time per Step: {stats['avg_time_per_step']:.3f}s")
    
    if stats['avg_time_per_epoch'] > 0:
        print(f"Average Time per Epoch: {stats['avg_time_per_epoch']:.3f}s")
    
    print("="*60)

# Example usage in a training script:
if __name__ == "__main__":
    print("Time Estimator for FAIRChem Training")
    print("="*50)
    print("This module provides TimeEstimationCallback that can be used")
    print("with FAIRChem's TrainEvalRunner for real-time time estimation.")
    print("\nTo use in your training script:")
    print("1. Import: from src_v2.time_estimator import TimeEstimationCallback")
    print("2. Create callback: time_callback = TimeEstimationCallback(...)")
    print("3. Add to callbacks list in TrainEvalRunner")
    print("4. Run training as usual")
    print("\nThe callback will automatically provide time estimates during training!")
