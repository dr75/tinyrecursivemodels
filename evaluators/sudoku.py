from typing import Dict, Optional
import torch
import torch.distributed as dist
from dataset.common import PuzzleDatasetMetadata


class Sudoku:
    """Evaluator for Sudoku puzzles - computes exact match accuracy."""
    required_outputs = {"preds", "labels", "puzzle_identifiers"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata):
        super().__init__()
        self.eval_metadata = eval_metadata
        self.data_path = data_path

        # Accumulators
        self.total_correct = 0
        self.total_puzzles = 0
        self.total_examples = 0  # Track total augmented examples
        self.puzzle_correct = {}  # Track per-puzzle correctness

    def begin_eval(self):
        """Reset counters at the start of evaluation."""
        self.total_correct = 0
        self.total_puzzles = 0
        self.total_examples = 0
        self.puzzle_correct = {}

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        """Process a batch of predictions."""
        predictions = preds["preds"]  # [batch, seq_len]
        labels = batch["labels"]  # [batch, seq_len]
        puzzle_ids = batch["puzzle_identifiers"]  # [batch]

        # Check exact match for each puzzle in the batch
        # Ignore padding (labels == -100)
        valid_mask = labels != -100
        correct_per_token = (predictions == labels) | ~valid_mask

        # A puzzle is correct if ALL tokens match
        correct_per_puzzle = correct_per_token.all(dim=1)  # [batch]

        # Count total examples processed (including augmentations)
        self.total_examples += len(puzzle_ids)

        # Accumulate per-puzzle results
        for puzzle_id, is_correct in zip(puzzle_ids.cpu().numpy(), correct_per_puzzle.cpu().numpy()):
            puzzle_id = int(puzzle_id)
            if puzzle_id not in self.puzzle_correct:
                self.puzzle_correct[puzzle_id] = []
            self.puzzle_correct[puzzle_id].append(bool(is_correct))

    def result(self, save_path: Optional[str] = None, rank: int = 0, world_size: int = 1, group: Optional[dist.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        """Compute final metrics."""

        # Aggregate per-puzzle: a puzzle is solved if ANY attempt was correct
        per_puzzle_solved = {
            puzzle_id: any(attempts)
            for puzzle_id, attempts in self.puzzle_correct.items()
        }

        local_correct = sum(per_puzzle_solved.values())
        local_total = len(per_puzzle_solved)

        # Reduce across ranks
        if world_size > 1 and group is not None:
            correct_tensor = torch.tensor([local_correct], dtype=torch.long)
            total_tensor = torch.tensor([local_total], dtype=torch.long)

            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM, group=group)

            global_correct = correct_tensor.item()
            global_total = total_tensor.item()
        else:
        if rank == 0:
            accuracy = global_correct / max(global_total, 1)
            return {
                "Sudoku/exact_match_accuracy": accuracy,
                "Sudoku/solved_puzzles": global_correct,
                "Sudoku/total_puzzles": global_total,
                "Sudoku/total_examples": self.total_examples,
            }

        return Nonedoku/total_puzzles": global_total,
            }

        return None
