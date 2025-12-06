#!/usr/bin/env python3
"""
Simplified training script for Tiny Recursive Models.

Usage:
    python train.py sudoku                    # Full training (requires GPU with 40GB+ VRAM)
    python train.py sudoku --small            # Laptop mode (works with 8GB memory, CPU)
    python train.py sudoku --no-attention     # Use MLP-T instead of attention
    python train.py maze --small
    python train.py arc-agi-1 --small
"""

import sys
import argparse
import subprocess
from pathlib import Path


# Dataset configurations: (data_path, epochs, eval_interval, lr, puzzle_emb_lr, weight_decay, H_cycles, L_cycles, extra_args)
DATASETS = {
    "sudoku": ("data/sudoku-extreme-1k-aug-1000", 50000, 5000, "1e-4", "1e-4", "1.0", 3, 6, {"evaluators": True}),
    "maze": ("data/maze-30x30-hard-1k", 50000, 5000, "1e-4", "1e-4", "1.0", 3, 4, {"global_batch_size": 128}),
    "arc-agi-1": ("data/arc1concept-aug-1000", 100000, 10000, "1e-4", "1e-2", "0.1", 3, 4, {"checkpoint_every_eval": True, "evaluators": True}),
    "arc-agi-2": ("data/arc2concept-aug-1000", 100000, 10000, "1e-4", "1e-2", "0.1", 3, 4, {"checkpoint_every_eval": True, "evaluators": True}),
}


def build_command(dataset: str, attention: bool, small: bool, tiny: bool, verbose: bool) -> list:
    """Build the pretrain.py command with appropriate arguments."""

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASETS.keys())}")

    data_path, epochs, eval_interval, lr, puzzle_emb_lr, weight_decay, H_cycles, L_cycles, extra = DATASETS[dataset]

    # Check if dataset exists
    if not Path(data_path).exists():
        print(f"Error: Dataset not found at {data_path}")
        print(f"Please prepare the dataset first. See README.md for instructions.")
        sys.exit(1)

    global_batch_size = extra.get("global_batch_size", None)
    hidden_size = None
    L_layers = 2
    max_test_batches = None

    # Adjust for small/tiny mode (lower memory, faster iterations)
    if small or tiny:
        epochs_orig = epochs
        epochs = min(epochs, 200)  # Cap epochs for quick testing
        eval_interval = 50
        H_cycles = 2  # Reduce computation
        L_cycles = 3
        global_batch_size = 32  # Much smaller batch size
        hidden_size = 256  # Smaller model
        L_layers = 2 if tiny else 1  # tiny has 2 layers, small has 1
        max_test_batches = 10  # Only evaluate on 10 batches
        mode_name = "tiny" if tiny else "small"
        print(f"🔧 {mode_name.capitalize()} mode enabled: epochs={epochs} (from {epochs_orig}), batch_size={global_batch_size}, hidden_size={hidden_size}, L_layers={L_layers}, max_test_batches={max_test_batches}")

    # Build command
    mode = "att" if attention else "mlp_t"
    run_suffix = "_tiny" if tiny else ("_small" if small else "")

    # Select evaluator based on dataset
    if extra.get("evaluators"):
        if dataset.startswith("arc-agi"):
            evaluator_config = "evaluators=[{name: arc@ARC}]"
        elif dataset == "sudoku":
            evaluator_config = "evaluators=[{name: sudoku@Sudoku}]"
        elif dataset == "maze":
            evaluator_config = "evaluators=[]"  # No maze evaluator yet
        else:
            evaluator_config = "evaluators=[]"
    else:
        evaluator_config = "evaluators=[]"

    cmd = [
        "python", "pretrain.py",
        "arch=trm",
        f"data_paths=[{data_path}]",
        evaluator_config,
        f"epochs={epochs}",
        f"eval_interval={eval_interval}",
        f"lr={lr}",
        f"puzzle_emb_lr={puzzle_emb_lr}",
        f"weight_decay={weight_decay}",
        f"puzzle_emb_weight_decay={weight_decay}",
        f"arch.L_layers={L_layers}",
        f"arch.H_cycles={H_cycles}",
        f"arch.L_cycles={L_cycles}",
        f"+run_name=pretrain_{mode}_{dataset}{run_suffix}",
        "ema=True",
    ]

    # Add batch size
    if global_batch_size:
        cmd.append(f"global_batch_size={global_batch_size}")

    # Add hidden size for small mode
    if hidden_size:
        cmd.append(f"arch.hidden_size={hidden_size}")

    # Add max test batches for small mode
    if max_test_batches:
        cmd.append(f"+max_test_batches={max_test_batches}")

    # Add verbose flag
    if verbose:
        cmd.append("+verbose=True")

    # Add MLP-T specific config
    if not attention:
        cmd.append("arch.mlp_t=True")
        if dataset == "sudoku":
            cmd.append("arch.pos_encodings=none")

    # Add optional parameters
    if extra.get("checkpoint_every_eval"):
        cmd.append("checkpoint_every_eval=True")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Train Tiny Recursive Models")
    parser.add_argument("dataset", choices=list(DATASETS.keys()), help="Dataset to train on")
    parser.add_argument("--no-attention", action="store_true", help="Use MLP-T instead of attention")
    parser.add_argument("--small", action="store_true", help="Laptop mode with 1 layer: batch_size=32, hidden_size=256, L_layers=1, fewer epochs")
    parser.add_argument("--tiny", action="store_true", help="Laptop mode with 2 layers: batch_size=32, hidden_size=256, L_layers=2, fewer epochs")
    parser.add_argument("--verbose", action="store_true", help="Print training and evaluation metrics to console")
    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.small and args.tiny:
        print("Error: --small and --tiny are mutually exclusive. Choose one.")
        sys.exit(1)

    # Build and run command
    cmd = build_command(args.dataset, attention=not args.no_attention, small=args.small, tiny=args.tiny, verbose=args.verbose)
    mode = "MLP-T" if args.no_attention else "attention"

    print(f"Training {args.dataset} with {mode}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, KeyboardInterrupt):
        sys.exit(1)


if __name__ == "__main__":
    main()
