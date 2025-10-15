#!/usr/bin/env python3
"""
N-Queens Hill-Climbing Search implementations and experiment runner.

Implements:
- Basic Hill-Climbing (complete-state formulation)
- Hill-Climbing with Sideways Moves (limit configurable; default 100)
- Random-Restart Hill-Climbing (with and without sideways)

Includes:
- Heuristic function: number of attacking queen pairs
- Board visualization (ASCII grid)
- Timing instrumentation
- Experiment harness to reproduce results and save CSVs
- Report generator (Markdown) that compiles experiment results into a PDF-ready document

Usage examples:
  python nqueens_hill_climbing.py run-experiments --n 8
  python nqueens_hill_climbing.py generate-report --n 8
  python nqueens_hill_climbing.py solve --n 8 --sideways --seed 123

All randomness is controlled by a Python random.Random instance. Use --seed for reproducibility.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ================================
# Global configuration variables
# ================================
# These are module-level constants that affect the algorithms and experiments.
# They can be overridden via CLI flags when relevant.
DEFAULT_SIDEWAYS_LIMIT: int = 100  # Maximum consecutive sideways moves in sideways hill-climbing
DEFAULT_RANDOM_RESTART_TRIALS: int = 200  # Trials for averaging random-restart metrics
RESULTS_DIR: str = os.path.join("results")
REPORTS_DIR: str = os.path.join("reports")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ================================
# Data classes for results
# ================================
@dataclass
class RunResult:
    """Outcome of a single hill-climbing run from an initial state."""
    success: bool
    steps: int
    time_ms: float
    final_state: List[int]
    final_h: int
    sequence: Optional[List[Tuple[List[int], int]]] = None  # [(state, h), ...]
    sideways_moves_used: int = 0


@dataclass
class TrialSummary:
    """Aggregated metrics across many runs (for a fixed number of trials)."""
    trials: int
    success_rate: float
    failure_rate: float
    avg_steps_success: float
    avg_steps_failure: float
    avg_time_success_ms: float
    avg_time_failure_ms: float


@dataclass
class RandomRestartSummary:
    label: str  # "without_sideways" or "with_sideways"
    trials: int
    avg_restarts: float
    avg_total_steps: float
    avg_total_time_ms: float


# ================================
# Core N-Queens utilities
# ================================

def compute_heuristic(state: List[int]) -> int:
    """Compute the number of attacking queen pairs.

    State representation: state[c] = row index of the queen in column c.

    Two queens attack each other if they share a row or a diagonal.
    This counts unordered pairs (i < j) that are in conflict.
    """
    h = 0
    n = len(state)
    for c1 in range(n):
        r1 = state[c1]
        for c2 in range(c1 + 1, n):
            r2 = state[c2]
            if r1 == r2:
                h += 1
            elif abs(r1 - r2) == abs(c1 - c2):
                h += 1
    return h


def generate_random_state(n: int, rng: random.Random) -> List[int]:
    """Generate a random complete state: one queen per column, random row assignment."""
    return [rng.randrange(n) for _ in range(n)]


def format_board(state: List[int]) -> str:
    """Return an ASCII visualization of the board for the given state."""
    n = len(state)
    lines: List[str] = []
    horizontal_border = "+" + "+".join(["---"] * n) + "+"
    lines.append(horizontal_border)
    for r in range(n):
        row_cells: List[str] = []
        for c in range(n):
            row_cells.append(" Q " if state[c] == r else "   ")
        lines.append("|" + "|".join(row_cells) + "|")
        lines.append(horizontal_border)
    return "\n".join(lines)


def enumerate_neighbors(state: List[int]) -> List[Tuple[List[int], int, Tuple[int, int]]]:
    """Generate all neighbors by moving one queen within its column to a different row.

    Returns a list of tuples: (neighbor_state, neighbor_h, move)
    where move = (column_index, new_row)
    """
    n = len(state)
    neighbors: List[Tuple[List[int], int, Tuple[int, int]]] = []
    for col in range(n):
        current_row = state[col]
        for new_row in range(n):
            if new_row == current_row:
                continue
            neighbor = state.copy()
            neighbor[col] = new_row
            h = compute_heuristic(neighbor)
            neighbors.append((neighbor, h, (col, new_row)))
    return neighbors


def choose_best_neighbors(state: List[int]) -> Tuple[List[List[int]], int]:
    """Return all neighbors with the minimal heuristic, and the minimal heuristic value."""
    neighbors = enumerate_neighbors(state)
    if not neighbors:
        return [], compute_heuristic(state)
    min_h = min(h for _, h, _ in neighbors)
    best = [s for (s, h, _) in neighbors if h == min_h]
    return best, min_h


# ================================
# Hill-Climbing algorithms
# ================================

def hill_climbing_search(
    initial_state: List[int],
    rng: random.Random,
    allow_sideways: bool = False,
    sideways_limit: int = DEFAULT_SIDEWAYS_LIMIT,
    record_sequence: bool = False,
    max_steps: Optional[int] = None,
) -> RunResult:
    """Perform hill-climbing from the given initial state.

    - If allow_sideways is False: only accept strictly better neighbors (lower h)
    - If allow_sideways is True: allow moves with equal h, up to sideways_limit consecutive times
    - record_sequence: if True, returns the sequence of (state, h) at each step including initial and final
    - max_steps: optional guard to stop after a large number of steps (defaults to N*N*10 when None)
    """
    n = len(initial_state)
    current = initial_state.copy()
    current_h = compute_heuristic(current)
    steps = 0
    sideways_used = 0
    start_time = time.perf_counter()

    if max_steps is None:
        max_steps = max(1000, n * n * 10)

    sequence: Optional[List[Tuple[List[int], int]]] = [] if record_sequence else None
    if record_sequence:
        sequence.append((current.copy(), current_h))

    while steps < max_steps and current_h > 0:
        best_neighbors, best_h = choose_best_neighbors(current)
        if not best_neighbors:
            break

        if best_h < current_h:
            # Strict improvement
            current = rng.choice(best_neighbors)
            current_h = best_h
            steps += 1
            sideways_used = 0
            if record_sequence:
                sequence.append((current.copy(), current_h))
            continue

        if allow_sideways and best_h == current_h:
            if sideways_used < sideways_limit:
                current = rng.choice(best_neighbors)
                current_h = best_h
                steps += 1
                sideways_used += 1
                if record_sequence:
                    sequence.append((current.copy(), current_h))
                continue
            else:
                # Ran out of sideways moves
                break

        # No improvement and not allowed to take sideways: local maximum
        break

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0
    success = current_h == 0

    return RunResult(
        success=success,
        steps=steps,
        time_ms=elapsed_ms,
        final_state=current,
        final_h=current_h,
        sequence=sequence,
        sideways_moves_used=sideways_used,
    )


def random_restart_hill_climbing(
    n: int,
    rng: random.Random,
    allow_sideways: bool = False,
    sideways_limit: int = DEFAULT_SIDEWAYS_LIMIT,
    max_steps_per_restart: Optional[int] = None,
) -> Tuple[int, int, float]:
    """Run hill-climbing with random restarts until a solution is found.

    Returns a tuple (restarts, total_steps, total_time_ms) for this one end-to-end run.

    - restarts: how many times we generated a new random state after a failure
      (0 if the first climb succeeded)
    - total_steps: total steps across all climbs
    - total_time_ms: total wall time across all climbs
    """
    restarts = 0
    total_steps = 0
    total_time_ms = 0.0

    while True:
        initial = generate_random_state(n, rng)
        rr = hill_climbing_search(
            initial_state=initial,
            rng=rng,
            allow_sideways=allow_sideways,
            sideways_limit=sideways_limit,
            record_sequence=False,
            max_steps=max_steps_per_restart,
        )
        total_steps += rr.steps
        total_time_ms += rr.time_ms
        if rr.success:
            break
        restarts += 1

    return restarts, total_steps, total_time_ms


# ================================
# Experiment harness
# ================================

def summarize_trials(
    n: int,
    rng: random.Random,
    trials: int,
    allow_sideways: bool,
    sideways_limit: int,
) -> TrialSummary:
    """Run many independent climbs and compute aggregate statistics."""
    success_steps: List[int] = []
    failure_steps: List[int] = []
    success_times_ms: List[float] = []
    failure_times_ms: List[float] = []

    for _ in range(trials):
        init_state = generate_random_state(n, rng)
        rr = hill_climbing_search(
            initial_state=init_state,
            rng=rng,
            allow_sideways=allow_sideways,
            sideways_limit=sideways_limit,
            record_sequence=False,
        )
        if rr.success:
            success_steps.append(rr.steps)
            success_times_ms.append(rr.time_ms)
        else:
            failure_steps.append(rr.steps)
            failure_times_ms.append(rr.time_ms)

    total = trials
    success_count = len(success_steps)
    failure_count = len(failure_steps)
    success_rate = (success_count / total) * 100.0
    failure_rate = (failure_count / total) * 100.0

    avg_steps_success = statistics.mean(success_steps) if success_steps else 0.0
    avg_steps_failure = statistics.mean(failure_steps) if failure_steps else 0.0
    avg_time_success_ms = statistics.mean(success_times_ms) if success_times_ms else 0.0
    avg_time_failure_ms = statistics.mean(failure_times_ms) if failure_times_ms else 0.0

    return TrialSummary(
        trials=trials,
        success_rate=success_rate,
        failure_rate=failure_rate,
        avg_steps_success=avg_steps_success,
        avg_steps_failure=avg_steps_failure,
        avg_time_success_ms=avg_time_success_ms,
        avg_time_failure_ms=avg_time_failure_ms,
    )


def write_trial_summaries_csv(
    path: str,
    summaries: List[TrialSummary],
) -> None:
    """Write summaries for different trial counts to CSV."""
    fieldnames = [
        "trials",
        "success_rate",
        "failure_rate",
        "avg_steps_success",
        "avg_steps_failure",
        "avg_time_success_ms",
        "avg_time_failure_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "trials": s.trials,
                "success_rate": f"{s.success_rate:.2f}",
                "failure_rate": f"{s.failure_rate:.2f}",
                "avg_steps_success": f"{s.avg_steps_success:.2f}",
                "avg_steps_failure": f"{s.avg_steps_failure:.2f}",
                "avg_time_success_ms": f"{s.avg_time_success_ms:.3f}",
                "avg_time_failure_ms": f"{s.avg_time_failure_ms:.3f}",
            })


def collect_sequences(
    n: int,
    rng: random.Random,
    allow_sideways: bool,
    sideways_limit: int,
    desired_successes: int = 2,
    desired_failures: int = 2,
    max_attempts: int = 200,
) -> Tuple[List[RunResult], List[RunResult]]:
    """Collect a set of complete sequences for report inclusion.

    Tries to collect at least desired_successes and desired_failures sequences.
    Returns (success_sequences, failure_sequences).
    """
    successes: List[RunResult] = []
    failures: List[RunResult] = []

    attempts = 0
    while attempts < max_attempts and (len(successes) < desired_successes or len(failures) < desired_failures):
        attempts += 1
        init_state = generate_random_state(n, rng)
        rr = hill_climbing_search(
            initial_state=init_state,
            rng=rng,
            allow_sideways=allow_sideways,
            sideways_limit=sideways_limit,
            record_sequence=True,
        )
        if rr.success and len(successes) < desired_successes:
            successes.append(rr)
        elif (not rr.success) and len(failures) < desired_failures:
            failures.append(rr)

    # If not enough of one type, top up with whatever we got
    all_collected = successes + failures
    if len(all_collected) < (desired_successes + desired_failures):
        # Try to add more runs regardless of success/failure
        while attempts < max_attempts and len(all_collected) < (desired_successes + desired_failures):
            attempts += 1
            init_state = generate_random_state(n, rng)
            rr = hill_climbing_search(
                initial_state=init_state,
                rng=rng,
                allow_sideways=allow_sideways,
                sideways_limit=sideways_limit,
                record_sequence=True,
            )
            all_collected.append(rr)
        # Re-split into successes/failures in order
        successes = [r for r in all_collected if r.success][:desired_successes]
        failures = [r for r in all_collected if not r.success][:desired_failures]

    return successes, failures


def write_sequences_file(path: str, sequences: List[RunResult], title: str) -> None:
    """Write a text file with detailed step-by-step sequences including ASCII boards."""
    with open(path, "w") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n\n")
        for idx, rr in enumerate(sequences, 1):
            f.write(f"Sequence {idx}: {'SUCCESS' if rr.success else 'FAILURE'}\n")
            f.write(f"Steps: {rr.steps}, Time: {rr.time_ms:.3f} ms, Final h: {rr.final_h}\n")
            f.write("Initial and intermediate states with h-values:\n\n")
            if rr.sequence is None:
                f.write("  [No sequence recorded]\n\n")
            else:
                for step_idx, (state, h) in enumerate(rr.sequence):
                    f.write(f"Step {step_idx:02d} | h={h}\n")
                    f.write(format_board(state) + "\n\n")
            f.write("-" * 40 + "\n\n")


def run_fixed_trial_sets(
    n: int,
    rng: random.Random,
    sideways_limit: int,
) -> Dict[str, Any]:
    """Run the required experiments for fixed trial sets and return paths to results."""
    trial_sets = [50, 100, 200, 500, 1000, 1500]

    # Basic Hill-Climbing
    basic_summaries: List[TrialSummary] = []
    print("Running Basic Hill-Climbing experiments...")
    for t in trial_sets:
        summary = summarize_trials(n, rng, t, allow_sideways=False, sideways_limit=sideways_limit)
        basic_summaries.append(summary)
        print(f"  Trials={t:4d} | Success={summary.success_rate:6.2f}% | AvgStepsSuc={summary.avg_steps_success:6.2f} | AvgStepsFail={summary.avg_steps_failure:6.2f}")
    basic_csv = os.path.join(RESULTS_DIR, "basic_hill_climbing_summary.csv")
    write_trial_summaries_csv(basic_csv, basic_summaries)

    # Sideways Hill-Climbing
    sideways_summaries: List[TrialSummary] = []
    print("\nRunning Hill-Climbing with Sideways Moves experiments...")
    for t in trial_sets:
        summary = summarize_trials(n, rng, t, allow_sideways=True, sideways_limit=sideways_limit)
        sideways_summaries.append(summary)
        print(f"  Trials={t:4d} | Success={summary.success_rate:6.2f}% | AvgStepsSuc={summary.avg_steps_success:6.2f} | AvgStepsFail={summary.avg_steps_failure:6.2f}")
    sideways_csv = os.path.join(RESULTS_DIR, "sideways_hill_climbing_summary.csv")
    write_trial_summaries_csv(sideways_csv, sideways_summaries)

    # Sequences for report inclusion (4 each)
    print("\nCollecting sequences for Basic Hill-Climbing...")
    basic_successes, basic_failures = collect_sequences(n, rng, allow_sideways=False, sideways_limit=sideways_limit)
    basic_sequences_path = os.path.join(RESULTS_DIR, "sequences_basic.txt")
    write_sequences_file(basic_sequences_path, basic_successes + basic_failures, "Basic Hill-Climbing: Sequences")

    print("Collecting sequences for Sideways Hill-Climbing...")
    side_successes, side_failures = collect_sequences(n, rng, allow_sideways=True, sideways_limit=sideways_limit)
    sideways_sequences_path = os.path.join(RESULTS_DIR, "sequences_sideways.txt")
    write_sequences_file(sideways_sequences_path, side_successes + side_failures, "Hill-Climbing with Sideways Moves: Sequences")

    # Random-Restart experiments
    print("\nRunning Random-Restart experiments (averaging)...")
    rr_trials = DEFAULT_RANDOM_RESTART_TRIALS

    def rr_summarize(label: str, allow_sw: bool) -> RandomRestartSummary:
        restarts_list: List[int] = []
        steps_list: List[int] = []
        time_list: List[float] = []
        for _ in range(rr_trials):
            restarts, total_steps, total_time_ms = random_restart_hill_climbing(
                n=n,
                rng=rng,
                allow_sideways=allow_sw,
                sideways_limit=sideways_limit,
            )
            restarts_list.append(restarts)
            steps_list.append(total_steps)
            time_list.append(total_time_ms)
        return RandomRestartSummary(
            label=label,
            trials=rr_trials,
            avg_restarts=statistics.mean(restarts_list),
            avg_total_steps=statistics.mean(steps_list),
            avg_total_time_ms=statistics.mean(time_list),
        )

    rr_no_sw = rr_summarize("without_sideways", False)
    rr_with_sw = rr_summarize("with_sideways", True)

    rr_csv = os.path.join(RESULTS_DIR, "random_restart_summary.csv")
    with open(rr_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "trials", "avg_restarts", "avg_total_steps", "avg_total_time_ms"])
        writer.writeheader()
        for row in [rr_no_sw, rr_with_sw]:
            writer.writerow({
                "label": row.label,
                "trials": row.trials,
                "avg_restarts": f"{row.avg_restarts:.2f}",
                "avg_total_steps": f"{row.avg_total_steps:.2f}",
                "avg_total_time_ms": f"{row.avg_total_time_ms:.3f}",
            })

    print("\nExperiment summaries saved to:")
    print(f"  {basic_csv}")
    print(f"  {sideways_csv}")
    print(f"  {rr_csv}")
    print(f"  {basic_sequences_path}")
    print(f"  {sideways_sequences_path}")

    return {
        "basic_csv": basic_csv,
        "sideways_csv": sideways_csv,
        "rr_csv": rr_csv,
        "basic_sequences": basic_sequences_path,
        "sideways_sequences": sideways_sequences_path,
    }


# ================================
# Report generator (Markdown)
# ================================

def read_csv_to_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def generate_report_markdown(
    n: int,
    basic_csv: str,
    sideways_csv: str,
    rr_csv: str,
    basic_sequences_path: str,
    sideways_sequences_path: str,
    out_path: str,
) -> None:
    """Compose a PDF-ready Markdown report including experiment results and analysis."""
    basic_rows = read_csv_to_rows(basic_csv)
    side_rows = read_csv_to_rows(sideways_csv)
    rr_rows = read_csv_to_rows(rr_csv)

    # Load sequences files
    with open(basic_sequences_path, "r") as f:
        basic_sequences_txt = f.read()
    with open(sideways_sequences_path, "r") as f:
        sideways_sequences_txt = f.read()

    def table_from_rows(rows: List[Dict[str, str]], headers: List[str]) -> str:
        header_line = "| " + " | ".join(headers) + " |\n"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |\n"
        lines = [header_line, sep_line]
        for r in rows:
            line = "| " + " | ".join(r.get(h, "") for h in headers) + " |\n"
            lines.append(line)
        return "".join(lines)

    # Build the Markdown content
    lines: List[str] = []
    lines.append("---")
    lines.append("title: N-Queens Hill-Climbing Search: Implementation, Experiments, and Analysis")
    lines.append("author: Auto-generated by nqueens_hill_climbing.py")
    lines.append("date: ")
    lines.append("---\n")

    lines.append("## Table of Contents\n")
    toc_items = [
        "1. Introduction & Problem Formulation",
        "2. Heuristic Function",
        "3. Program Structure",
        "4. Hill-Climbing Algorithms",
        "5. Experimental Results",
        "6. Analysis & Discussion",
        "7. Conclusion",
        "8. Appendix",
    ]
    for item in toc_items:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 1. Introduction & Problem Formulation\n")
    lines.append(
        "The N-Queens problem asks for a placement of N queens on an N×N chessboard such that no two queens "
        "attack each other (no shared rows, columns, or diagonals). We use a complete-state formulation: "
        "each column contains exactly one queen, and a state is a length-N array whose c-th element gives "
        "the row of the queen in column c. The goal is any state with heuristic value 0. The path cost is "
        "irrelevant since we only care about reaching any goal state via local search.\n"
    )

    lines.append("## 2. Heuristic Function\n")
    lines.append(
        "We use the standard heuristic h(s) = number of attacking pairs of queens in state s. "
        "Mathematically, for columns i < j with rows r_i and r_j, a conflict occurs if r_i = r_j (same row) "
        "or |r_i − r_j| = |i − j| (same diagonal). h(s) is the count of such pairs. Example for N=4: if state = [1, 3, 0, 2], "
        "h(s) = 0 because no two queens share a row or diagonal.\n"
    )

    lines.append("## 3. Program Structure\n")
    lines.append("Text-based architecture diagram:\n")
    lines.append("""
    nqueens_hill_climbing.py
    ├── Core utilities
    │   ├── compute_heuristic(state)
    │   ├── generate_random_state(n, rng)
    │   ├── format_board(state)
    │   └── enumerate_neighbors(state) / choose_best_neighbors(state)
    ├── Algorithms
    │   ├── hill_climbing_search(... allow_sideways)
    │   └── random_restart_hill_climbing(... allow_sideways)
    ├── Experiments
    │   ├── summarize_trials(...)
    │   ├── collect_sequences(...)
    │   └── run_fixed_trial_sets(...)
    └── Reporting
        ├── write_trial_summaries_csv(...)
        └── generate_report_markdown(...)
    """.strip() + "\n")

    lines.append("Global variables:\n")
    lines.append("- DEFAULT_SIDEWAYS_LIMIT: maximum consecutive sideways moves (default 100)\n")
    lines.append("- DEFAULT_RANDOM_RESTART_TRIALS: trials to average RR metrics (default 200)\n")
    lines.append(f"- RESULTS_DIR: '{RESULTS_DIR}' output path\n")
    lines.append(f"- REPORTS_DIR: '{REPORTS_DIR}' output path\n")

    lines.append("Key functions and signatures (purpose → inputs → outputs):\n")
    lines.append("- compute_heuristic(state: List[int]) → int: number of attacking pairs\n")
    lines.append("- generate_random_state(n: int, rng: Random) → List[int]: random complete state\n")
    lines.append("- format_board(state: List[int]) → str: ASCII board visualization\n")
    lines.append("- enumerate_neighbors(state) → List[(state, h, move)]: single-column moves\n")
    lines.append("- choose_best_neighbors(state) → (best_states, best_h): neighbors with minimal h\n")
    lines.append("- hill_climbing_search(initial_state, rng, allow_sideways, sideways_limit, record_sequence, max_steps) → RunResult\n")
    lines.append("- random_restart_hill_climbing(n, rng, allow_sideways, sideways_limit, max_steps_per_restart) → (restarts, steps, time_ms)\n")
    lines.append("- summarize_trials(...) → TrialSummary: aggregate metrics over many runs\n")
    lines.append("- collect_sequences(...) → success/failure sequences for report\n")
    lines.append("- run_fixed_trial_sets(...) → CSV and sequence file outputs\n")

    lines.append("## 4. Hill-Climbing Algorithms\n")
    lines.append("Basic Hill-Climbing (no sideways moves):\n")
    lines.append("""
```
function HILL-CLIMBING(state):
  loop:
    best_neighbors ← neighbors with minimal h
    if min_h < h(state):
      state ← random choice(best_neighbors)
    else:
      return state  // local maximum or goal
```
""".strip() + "\n")

    lines.append("Hill-Climbing with Sideways Moves (limit S):\n")
    lines.append("""
```
function HILL-CLIMBING-SIDEWAYS(state, S):
  sideways ← 0
  loop:
    best_neighbors ← neighbors with minimal h
    if min_h < h(state):
      state ← random choice(best_neighbors); sideways ← 0
    else if min_h = h(state) and sideways < S:
      state ← random choice(best_neighbors); sideways ← sideways + 1
    else:
      return state
```
""".strip() + "\n")

    lines.append("Random-Restart (using a chosen base climber):\n")
    lines.append("""
```
function RANDOM-RESTART-CLIMB(N, climber):
  restarts ← 0; total_steps ← 0
  loop:
    state ← RANDOM-STATE(N)
    result ← climber(state)
    total_steps ← total_steps + result.steps
    if result.success: return (restarts, total_steps)
    restarts ← restarts + 1
```
""".strip() + "\n")

    lines.append("## 5. Experimental Results\n")
    lines.append("### Section A: Basic Hill-Climbing\n")
    lines.append("Trials vs metrics:\n")
    lines.append(table_from_rows(basic_rows, [
        "trials",
        "success_rate",
        "failure_rate",
        "avg_steps_success",
        "avg_steps_failure",
    ]) + "\n")

    lines.append("Four complete search sequences (initial → steps → final):\n")
    lines.append("""
```
""".strip() + "\n")
    lines.append(basic_sequences_txt)
    lines.append("""
```
""".strip() + "\n")

    lines.append("### Section B: Hill-Climbing with Sideways Moves\n")
    lines.append("Trials vs metrics:\n")
    lines.append(table_from_rows(side_rows, [
        "trials",
        "success_rate",
        "failure_rate",
        "avg_steps_success",
        "avg_steps_failure",
    ]) + "\n")

    lines.append("Four complete search sequences (initial → steps → final):\n")
    lines.append("""
```
""".strip() + "\n")
    lines.append(sideways_sequences_txt)
    lines.append("""
```
""".strip() + "\n")

    lines.append("### Section C: Random-Restart Hill-Climbing\n")
    lines.append("Comparison with and without sideways moves:\n")
    lines.append(table_from_rows(rr_rows, [
        "label",
        "trials",
        "avg_restarts",
        "avg_total_steps",
        "avg_total_time_ms",
    ]) + "\n")

    lines.append("## 6. Analysis & Discussion\n")
    lines.append(
        "Basic hill-climbing is fast but often gets stuck in local maxima or plateaus. "
        "Allowing sideways moves helps traverse plateaus, sometimes improving success rates, but may also "
        "linger without progress if the sideways limit is too high. Random restarts transform local search "
        "into an effective global search by repeatedly sampling different basins of attraction; combining "
        "this with sideways moves typically reduces expected restarts and total steps.\n"
    )

    lines.append("## 7. Conclusion\n")
    lines.append(
        "Among the tested approaches, random-restart hill-climbing with a moderate sideways-move limit "
        "achieves the highest reliability for N=8, with competitive total steps and time. This illustrates "
        "a key property of local search: strategic randomness can overcome local optima efficiently.\n"
    )

    lines.append("## 8. Appendix\n")
    lines.append("- Full source code resides in `nqueens_hill_climbing.py` with docstrings and comments.\n")
    lines.append("- The sequences files include ASCII board visualizations to illustrate conflicts and resolutions.\n")
    lines.append("- Raw CSV outputs are available in the `results/` directory for independent analysis.\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ================================
# CLI
# ================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N-Queens Hill-Climbing experiments and report")
    sub = parser.add_subparsers(dest="command", required=True)

    # solve command
    p_solve = sub.add_parser("solve", help="Solve a single instance and print the sequence")
    p_solve.add_argument("--n", type=int, default=8, help="Board size (number of queens)")
    p_solve.add_argument("--sideways", action="store_true", help="Allow sideways moves")
    p_solve.add_argument("--sideways-limit", type=int, default=DEFAULT_SIDEWAYS_LIMIT, help="Sideways move limit")
    p_solve.add_argument("--seed", type=int, default=42, help="Random seed")

    # run-experiments command
    p_run = sub.add_parser("run-experiments", help="Run all required experiments and save CSVs")
    p_run.add_argument("--n", type=int, default=8, help="Board size (number of queens)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.add_argument("--sideways-limit", type=int, default=DEFAULT_SIDEWAYS_LIMIT, help="Sideways move limit")
    p_run.add_argument("--rr-trials", type=int, default=DEFAULT_RANDOM_RESTART_TRIALS, help="Random-restart averaging trials")

    # generate-report command
    p_rep = sub.add_parser("generate-report", help="Generate Markdown report from existing results")
    p_rep.add_argument("--n", type=int, default=8, help="Board size (number of queens)")
    p_rep.add_argument("--basic-csv", type=str, default=os.path.join(RESULTS_DIR, "basic_hill_climbing_summary.csv"))
    p_rep.add_argument("--sideways-csv", type=str, default=os.path.join(RESULTS_DIR, "sideways_hill_climbing_summary.csv"))
    p_rep.add_argument("--rr-csv", type=str, default=os.path.join(RESULTS_DIR, "random_restart_summary.csv"))
    p_rep.add_argument("--basic-seq", type=str, default=os.path.join(RESULTS_DIR, "sequences_basic.txt"))
    p_rep.add_argument("--sideways-seq", type=str, default=os.path.join(RESULTS_DIR, "sequences_sideways.txt"))
    p_rep.add_argument("--out", type=str, default=os.path.join(REPORTS_DIR, "nqueens_hc_report.md"))

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.command == "solve":
        rng = random.Random(args.seed)
        init = generate_random_state(args.n, rng)
        print("Initial state (h=%d):" % compute_heuristic(init))
        print(format_board(init))
        result = hill_climbing_search(
            initial_state=init,
            rng=rng,
            allow_sideways=args.sideways,
            sideways_limit=args.sideways_limit,
            record_sequence=True,
        )
        print("\nRun complete:")
        print(f"  Success: {result.success}")
        print(f"  Steps: {result.steps}")
        print(f"  Time: {result.time_ms:.3f} ms")
        print(f"  Final h: {result.final_h}")
        if result.sequence:
            for idx, (state, h) in enumerate(result.sequence):
                print(f"\nStep {idx:02d} | h={h}")
                print(format_board(state))

    elif args.command == "run-experiments":
        # Allow override for RR trials via CLI by temporarily setting global default
        global DEFAULT_RANDOM_RESTART_TRIALS
        DEFAULT_RANDOM_RESTART_TRIALS = args.rr_trials

        rng = random.Random(args.seed)
        outputs = run_fixed_trial_sets(n=args.n, rng=rng, sideways_limit=args.sideways_limit)

        # Also generate report after experiments for convenience
        report_path = os.path.join(REPORTS_DIR, "nqueens_hc_report.md")
        generate_report_markdown(
            n=args.n,
            basic_csv=outputs["basic_csv"],
            sideways_csv=outputs["sideways_csv"],
            rr_csv=outputs["rr_csv"],
            basic_sequences_path=outputs["basic_sequences"],
            sideways_sequences_path=outputs["sideways_sequences"],
            out_path=report_path,
        )
        print(f"\nReport written to: {report_path}")

    elif args.command == "generate-report":
        generate_report_markdown(
            n=args.n,
            basic_csv=args.basic_csv,
            sideways_csv=args.sideways_csv,
            rr_csv=args.rr_csv,
            basic_sequences_path=args.basic_seq,
            sideways_sequences_path=args.sideways_seq,
            out_path=args.out,
        )
        print(f"Report written to: {args.out}")


if __name__ == "__main__":
    main()
