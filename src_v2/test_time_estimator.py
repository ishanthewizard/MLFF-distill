#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for TimeEstimationCallback
======================================

This script tests the time estimator to ensure it works correctly.
"""

import time
import logging
from src_v2.time_estimator import TimeEstimationCallback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_time_estimator():
    """Test the time estimator with mock data."""
    
    print("Testing TimeEstimationCallback...")
    
    # Create a mock dataloader (just need length)
    class MockDataloader:
        def __init__(self, length):
            self.length = length
        
        def __len__(self):
            return self.length
    
    # Create mock dataloaders
    train_dl = MockDataloader(100)  # 100 batches per epoch
    eval_dl = MockDataloader(20)    # 20 evaluation batches
    
    # Create time estimator
    time_callback = TimeEstimationCallback(
        update_interval=10,           # Update every 10 steps
        log_format="detailed",        # Detailed output
        max_epochs=10,                # 10 epochs
        max_steps=1000,               # 1000 steps max
        train_dataloader=train_dl,    # Training dataloader
        eval_dataloader=eval_dl,      # Evaluation dataloader
        evaluate_every_n_steps=100    # Evaluate every 100 steps
    )
    
    print(f"Time estimator created successfully!")
    print(f"Total steps calculated: {time_callback.total_steps}")
    print(f"Total epochs: {time_callback.total_epochs}")
    
    # Simulate some training progress
    print("\nSimulating training progress...")
    
    # Simulate training start
    time_callback.on_train_start(None, None)
    
    # Simulate a few steps
    for step in range(0, 50, 10):
        time_callback.current_step = step
        time_callback.current_epoch = step // 100
        
        # Simulate step timing
        if step > 0:
            time_callback.step_times.append(0.1)  # 0.1 seconds per step
        
        # Update time estimates
        if step % 10 == 0 and step > 0:
            time_callback._update_time_estimates()
        
        time.sleep(0.1)  # Small delay to simulate real training
    
    # Get final stats
    stats = time_callback.get_current_stats()
    print("\nFinal statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_time_estimator()
