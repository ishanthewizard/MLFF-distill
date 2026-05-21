#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Compilation Wrapper for FAIRChem
======================================

This module provides a callback to enable torch.compile for FAIRChem models
without modifying the fairchem source code.

IMPORTANT: This callback ensures that checkpoints save the original
uncompiled model, so checkpoints remain fully compatible with standard
inference and can be loaded normally. The compilation is only used during
training for performance optimization.

Usage:
    from src_v2.model_compile_wrapper import CompileCallback
    
    callbacks = [
        CompileCallback(compile_mode="reduce-overhead"),  # or "max-autotune", "default"
        # ... other callbacks
    ]
    
    runner = TrainEvalRunner(
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        train_eval_unit=unit,  # Use normal MLIPTrainEvalUnit
        callbacks=callbacks,
        ...
    )

Or via config:
    runner:
      callbacks:
        - _target_: src_v2.model_compile_wrapper.CompileCallback
          compile_mode: reduce-overhead
          dynamic: true
"""

import logging
import torch
from typing import Literal
from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit


class CompileCallback(Callback):
    """
    A callback that compiles the model using torch.compile at training start.
    
    This approach compiles the model after DDP/FSDP wrapping is complete,
    which is the recommended approach for distributed training.
    
    IMPORTANT: This callback ensures that checkpoints save the original
    uncompiled model, so checkpoints remain compatible with non-compiled
    inference and can be loaded normally.
    
    Args:
        compile_mode: Compilation mode for torch.compile. Options:
            - "default": Standard compilation
            - "reduce-overhead": Optimize for reduced overhead (recommended for training)
            - "max-autotune": Maximum optimization (slower compilation, faster runtime)
        fullgraph: If True, compile the entire graph (may fail for dynamic graphs)
        dynamic: If True, enable dynamic shapes (useful for variable-size inputs)
        backend: Backend to use (default: "inductor")
        disable: If True, disable compilation (useful for debugging)
    """
    
    def __init__(
        self,
        compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
        backend: str = "inductor",
        disable: bool = False,
    ):
        self.compile_mode = compile_mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.backend = backend
        self.disable = disable
        self._compiled = False
        self._original_model = None  # Store original model for checkpoint saving
        self._compiled_model = None  # Store compiled model
        self._unit = None  # Store reference to unit
        
    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        """Compile the model when training starts (after DDP wrapping)."""
        if self.disable:
            logging.info("Model compilation disabled by user")
            return
            
        if self._compiled:
            logging.warning("Model already compiled, skipping")
            return
            
        if not hasattr(unit, 'model'):
            logging.warning("Unit does not have a 'model' attribute, cannot compile")
            return
            
        model = unit.model
        if model is None:
            logging.warning("Model is None, cannot compile")
            return
        
        # Check if model is already compiled
        if hasattr(model, '_orig_mod'):
            logging.info("Model appears to already be compiled")
            self._compiled = True
            return
        
        try:
            logging.info(f"Compiling model with mode='{self.compile_mode}', "
                        f"fullgraph={self.fullgraph}, dynamic={self.dynamic}, "
                        f"backend='{self.backend}'")
            
            # Store original model and unit reference
            self._original_model = model
            self._unit = unit
            
            # Compile the model
            compiled_model = torch.compile(
                model,
                mode=self.compile_mode,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
                backend=self.backend,
            )
            
            # Store compiled model and replace in unit
            self._compiled_model = compiled_model
            unit.model = compiled_model
            self._compiled = True
            
            # Wrap save_state to swap models during checkpoint saving
            self._wrap_save_state(unit)
            
            logging.info("Model compilation successful!")
            logging.info("Checkpoints will save the original uncompiled model")
            
        except Exception as e:
            logging.error(f"Failed to compile model: {e}")
            logging.error("Training will continue without compilation")
            # Don't raise - allow training to continue without compilation
    
    def _wrap_save_state(self, unit: TTrainUnit) -> None:
        """
        Wrap the unit's save_state method to swap models during saving.
        
        This ensures checkpoints contain the original uncompiled model,
        making them compatible with standard inference and loading.
        The compiled model is only used during training for performance.
        """
        original_save_state = unit.save_state
        
        def save_state_with_model_swap(checkpoint_location: str) -> None:
            """
            Temporarily swap to original model, save, then restore compiled model.
            
            This ensures the checkpoint contains the original model (possibly DDP-wrapped)
            without the torch.compile wrapper, making it fully compatible with
            standard model loading and inference.
            """
            if self._compiled and self._original_model is not None:
                # Temporarily swap to original model for checkpoint saving
                logging.debug("Swapping to original model for checkpoint saving")
                unit.model = self._original_model
                try:
                    # Call original save_state - this will save the original model
                    original_save_state(checkpoint_location)
                finally:
                    # Always restore compiled model after saving for continued training
                    unit.model = self._compiled_model
                    logging.debug("Restored compiled model after checkpoint save")
            else:
                # Not compiled, just call original
                original_save_state(checkpoint_location)
        
        # Replace the save_state method
        unit.save_state = save_state_with_model_swap
    
    def on_train_step_start(self, state: State, unit: TTrainUnit) -> None:
        """Called before each training step - ensure model is compiled."""
        # This ensures the model stays compiled even if something tries to swap it
        if self._compiled and self._compiled_model is not None:
            if unit.model is not self._compiled_model:
                # Model was swapped (e.g., during checkpoint save), restore it
                if unit.model is self._original_model:
                    # This is expected after checkpoint save, restore compiled
                    unit.model = self._compiled_model
