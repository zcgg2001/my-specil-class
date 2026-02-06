# PyTorch iJIT_NotifyEvent Error Fix Summary

## Problem
The experiments E4, E5, and E6 were failing with the error:
```
ImportError: /home/lthpc/miniconda3/envs/smoke-class/lib/python3.10/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

This error occurs when PyTorch's shared library conflicts with Intel's JIT compiler (ITT) libraries, typically in HPC environments or when Intel profiling tools are loaded.

## Root Cause
The `iJIT_NotifyEvent` symbol is part of Intel's Instrumentation and Tracing Technology (ITT) API, used for performance profiling. When Intel profiling libraries are loaded into the environment, they can conflict with PyTorch's internal JIT compilation, causing the symbol resolution to fail.

## Solution Implemented

### 1. Created Safe Import Utility
**File**: `src/utils/safe_import.py`
- Implements environment variable cleanup before PyTorch import
- Removes Intel JIT/profiling related environment variables that cause conflicts
- Provides fallback mechanism if the primary import fails
- Re-exports commonly used torch submodules for convenience

### 2. Updated Project Files
Updated imports in the following files to use the safe import utility:

**Training Scripts:**
- `train_multitask.py`
- `train_cvae.py`
- `train_predictor.py`

**Source Code:**
- `src/models/multitask.py`
- `src/models/cvae.py`
- `src/models/predictor.py`
- `src/losses/physics_loss.py`
- `src/data/augmentor.py`
- `test_models.py`

The changes ensure that all modules importing PyTorch do so safely, preventing conflicts regardless of which module is imported first.

### 3. Fixed Minor Issues
- Fixed `model_name` scope issue in training loop
- Fixed Unicode encoding issue with R² symbol
- Improved model saving logic for short training runs

## Environment Variables Handled
The fix unsets the following problematic environment variables:
- `INTEL_LIBITTNOTIFY64`
- `INTEL_JIT_PROFILER64`
- `ITTNOTIFY_LD_PRELOAD`
- `LD_PRELOAD` (when it contains Intel libraries)
- `INTEL_PROFILE`
- `VTUNE_PROFILER_DIR`
- `AMPLXE_RUN_DIR`
- `ITT_DOMAIN`

## Testing Results
All previously failing experiments now work successfully:

✅ **E4**: split=random, augment=cvae_physics
- Macro-F1: 0.8750, RMSE: 0.8697

✅ **E5**: split=batch, augment=none
- Macro-F1: 0.8940, RMSE: 0.9781

✅ **E6**: split=batch, augment=cvae_physics
- Macro-F1: 0.7857, RMSE: 1.0548

## Usage
The fix is transparent to users. Simply run the training scripts as before:
```bash
python train_multitask.py --epochs 100 --split random --augment cvae_physics
```

The safe import utility automatically handles the environment cleanup without any user intervention required.

## Prevention
This fix prevents the iJIT_NotifyEvent error from occurring in environments where:
- Intel profiling tools are installed
- HPC clusters with Intel libraries are used
- Environment variables conflict with PyTorch's JIT compilation
- Multiple users share the same environment with different tool configurations