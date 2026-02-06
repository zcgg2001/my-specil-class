"""
Safe PyTorch import utility to handle Intel JIT compiler conflicts

This module provides a safe way to import PyTorch when encountering
the 'undefined symbol: iJIT_NotifyEvent' error caused by Intel profiling
tools and JIT compiler conflicts.
"""

import os
import sys

def safe_torch_import():
    """
    Safely import PyTorch by unsetting Intel JIT-related environment variables
    that can cause the 'undefined symbol: iJIT_NotifyEvent' error.

    Returns:
        module: The torch module

    Raises:
        ImportError: If PyTorch cannot be imported even after cleanup
    """
    # Store original environment for debugging
    original_env = {}

    # List of Intel JIT/profiling related environment variables that can cause conflicts
    intel_conflict_vars = [
        'INTEL_LIBITTNOTIFY64',
        'INTEL_JIT_PROFILER64',
        'ITTNOTIFY_LD_PRELOAD',
        'LD_PRELOAD',  # Sometimes contains Intel libraries
        'INTEL_PROFILE',
        'VTUNE_PROFILER_DIR',
        'AMPLXE_RUN_DIR',
        'ITT_DOMAIN'
    ]

    # Unset problematic environment variables
    for var in intel_conflict_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    try:
        import torch
        return torch
    except ImportError as e:
        # Restore original environment for debugging
        for var, value in original_env.items():
            os.environ[var] = value

        if "iJIT_NotifyEvent" in str(e):
            raise ImportError(
                f"Failed to import PyTorch due to Intel JIT compiler conflict: {e}\n"
                "This typically occurs in HPC environments with Intel profiling tools.\n"
                "Try running with a clean environment or contact your system administrator."
            ) from e
        else:
            raise

# Import torch using the safe method
torch = safe_torch_import()

# Re-export commonly used torch submodules for convenience
if hasattr(torch, 'nn'):
    nn = torch.nn
if hasattr(torch, 'optim'):
    optim = torch.optim

__all__ = ['torch', 'nn', 'optim', 'safe_torch_import']