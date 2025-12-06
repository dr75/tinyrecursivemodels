# Tiny Recursive Models - AI Agent Instructions

## Project Overview

This is **Tiny Recursion Model (TRM)**, a research codebase for recursive reasoning on puzzle-solving tasks (ARC-AGI, Sudoku, Maze). TRM achieves 45% on ARC-AGI-1 using only 7M parameters through **iterative answer refinement** with cycles of latent reasoning (H_cycles) and answer updates (L_cycles). The core innovation is simplifying recursive reasoning to its essence: recursively update latent state `z`, then update answer `y`, repeat.

Key architectural components:
- **models/recursive_reasoning/trm.py**: Main TRM model with ACT (Adaptive Computation Time) halting
- **models/recursive_reasoning/hrm.py**: Hierarchical Reasoning Model baseline (more complex, hierarchical approach)
- **puzzle_dataset.py**: Unified dataset interface for all puzzle types with augmentation support
- **pretrain.py**: Main training script with distributed training support

## Configuration System (Hydra)

All experiments are configured via **Hydra** with hierarchical YAML configs:
- `config/cfg_pretrain.yaml`: Base training config (data paths, hyperparams, evaluators)
- `config/arch/*.yaml`: Model architectures (trm, hrm, transformers_baseline, etc.)
  - Each defines `name: module.path@ClassName` for dynamic loading
  - Example: `recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1`

Override config from CLI:
```bash
python pretrain.py arch=trm data_paths="[data/sudoku]" lr=1e-4 arch.H_cycles=3
```

## Critical Training Patterns

### 1. Dual Optimizer Setup
TRM uses **two separate optimizers** with different learning rates:
- `CastedSparseEmbeddingSignSGD_Distributed`: For puzzle embeddings (per-puzzle learned vectors)
  - Operates on `model.model.puzzle_emb.buffers()` - note the sparse embedding uses buffers, not parameters
  - Custom SignSGD with all-reduce for distributed training
- `AdamATan2`: For model weights (AdamW variant with atan2-based adaptive learning)

The `puzzle_emb_lr` is typically 10-100x higher than base `lr` (e.g., 1e-2 vs 1e-4).

### 2. Sparse Puzzle Embeddings
Each puzzle gets a unique learned embedding via `CastedSparseEmbedding`:
- Training: Copies relevant embeddings to `local_weights` buffer, computes gradients, then aggregates back to main `weights`
- Inference: Direct lookup from `weights`, no gradient tracking
- **Critical**: The embedding dimension can differ from model dimension - padded/truncated to fit via `puzzle_emb_len`

### 3. ACT (Adaptive Computation Time) Loss
Model learns when to halt via Q-learning:
- `q_halt_logits`: Binary classifier predicting if current answer is correct
- `q_continue_logits`: Bootstrap target for continuing (disabled by default with `no_ACT_continue: True`)
- Loss components: `lm_loss` (cross-entropy) + `q_halt_loss` (binary halting) + optional `q_continue_loss`

### 4. Distributed Training
Uses PyTorch DDP patterns:
- Manually broadcasts parameters from rank 0 after initialization
- Manual `all_reduce` on gradients in `train_batch()` instead of DDP wrapper
- Per-GPU batch size: `global_batch_size // world_size`
- Launch with `torchrun`: Set `RANK`, `WORLD_SIZE`, `LOCAL_RANK` environment variables

### 5. Model Compilation & Checkpointing
- `torch.compile()` is default (disable with `DISABLE_COMPILE=1`)
- Checkpoints saved as state_dicts at `checkpoint_path/step_{N}`
- **Checkpoint loading handles puzzle embedding resizing** - if shape mismatch, resets to mean of old embeddings
- Compiled model keys prefixed with `_orig_mod.` in state_dict

## Dataset Preparation

All datasets must be pre-processed to standardized format:

```bash
# ARC-AGI with dihedral augmentation (8 symmetries)
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug 1000

# Sudoku/Maze
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

**Output structure**: Each dataset folder contains:
- `metadata.json`: Vocab size, sequence length, padding IDs, number of puzzles
- `identifiers.json`: Maps puzzle IDs to names
- `train.npz`, `test.npz`: Packed sequences with `examples`, `puzzle_indices`, `group_indices`
- `test_puzzles.json`: Original test examples for evaluation

## Model Architecture Conventions

### Layer Naming
- `L_level`: "Lower level" - operates on latent state `z_L`, updates answer
- `H_level`: "Higher level" (HRM only) - hierarchical latent `z_H`
- TRM simplifies by using only one latent level with recursive updates

### Custom Layers
- `SwiGLU`: Gated FFN activation (hidden_size → expansion*hidden_size → hidden_size)
- `CastedLinear/CastedEmbedding`: Auto-casting to `forward_dtype` (bfloat16 default)
- `rms_norm`: RMS normalization, **post-norm** architecture (norm after residual)
- `mlp_t`: Optional flag to replace attention with MLP over sequence dimension

### Position Encodings
Config `pos_encodings` controls: `"rope"` (RoPE, default), `"learned"`, `"none"`

## Evaluation System

Evaluators are registered in config and run during eval intervals:
```yaml
evaluators:
  - name: arc@ARC  # module.path@ClassName
    submission_K: 2
    pass_Ks: [1, 2, 5, 10]
```

Each evaluator:
- Declares `required_outputs` (e.g., `{"preds", "q_halt_logits", "puzzle_identifiers"}`)
- Implements `begin_eval()`, `update_batch()`, `result()` lifecycle
- ARC evaluator aggregates predictions with inverse augmentation and voting

## Development Workflow

### Running Experiments
```bash
# Single GPU
python pretrain.py arch=trm data_paths="[data/sudoku]" +run_name=my_experiment

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 pretrain.py arch=trm data_paths="[data/arc1]"

# Resume from checkpoint
python pretrain.py load_checkpoint=checkpoints/my_project/my_run/step_10000
```

### Debugging Tips
- Use `DISABLE_COMPILE=1` to disable torch.compile for better stack traces
- Check WandB logs for metrics: `train/accuracy`, `train/exact_accuracy`, `train/steps`, `ARC/pass@K`
- EMA (Exponential Moving Average) can be enabled with `ema=True ema_rate=0.999` for smoother convergence

### Adding New Models
1. Create `models/recursive_reasoning/my_model.py` with `MyModel_Inner` and config class
2. Add architecture YAML in `config/arch/my_model.yaml`:
   ```yaml
   name: recursive_reasoning.my_model@MyModel_ACTV1
   loss:
     name: losses@ACTLossHead
     loss_type: stablemax_cross_entropy
   # ... hyperparams
   ```
3. Models must implement `initial_carry()` and `forward()` returning `(new_carry, outputs)`

## Common Gotchas

- **`stablemax_cross_entropy`**: Custom numerically stable softmax (not standard PyTorch)
- **Non-autoregressive**: All models are parallel (no causal masking) - predict full output sequence at once
- **Ignore label**: Use `IGNORE_LABEL_ID = -100` for padding, not just for loss masking
- **Dihedral transforms**: Augmentations use 8 symmetries; evaluator must **inverse transform** predictions back
- **WandB logging**: Run `wandb login` before training to sync metrics
