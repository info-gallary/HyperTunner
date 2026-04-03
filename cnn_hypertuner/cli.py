#!/usr/bin/env python3
"""
run_tuner.py — Quick-start script for CNN HyperTuner.
Supports YAML config or direct CLI arguments.

Usage:
    python run_tuner.py                          # PSO on CIFAR-10 (defaults)
    python run_tuner.py --algorithm ga           # GA
    python run_tuner.py --algorithm sa           # SA
    python run_tuner.py --compare                # Compare all three
    python run_tuner.py --config configs/default.yaml
"""

import argparse
import sys
import os

# ── allow running from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="CNN HyperTuner — Quick Start")
    parser.add_argument("--algorithm", "-a", default="pso",
                        choices=["ga", "pso", "sa"],
                        help="Algorithm to use (default: pso)")
    parser.add_argument("--dataset", "-d", default="cifar10",
                        choices=["cifar10", "mnist"])
    parser.add_argument("--iterations", "-i", type=int, default=20)
    parser.add_argument("--epochs", "-e", type=int, default=3)
    parser.add_argument("--compare", action="store_true",
                        help="Compare GA vs PSO vs SA")
    parser.add_argument("--plot", action="store_true",
                        help="Show convergence plot after tuning")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from cnn_hypertuner import CNNHyperTuner

    if args.compare:
        print("\n🔬 Running Algorithm Comparison: GA vs PSO vs SA\n")
        results = CNNHyperTuner.compare_algorithms(
            algorithms=["ga", "pso", "sa"],
            dataset=args.dataset,
            n_iterations=args.iterations,
            n_epochs_per_eval=args.epochs,
            seed=args.seed,
        )
        if args.plot:
            from cnn_hypertuner.utils.plotting import plot_comparison
            plot_comparison(results, save_path="experiments/comparison/convergence_comparison.png")
    else:
        tuner = CNNHyperTuner(
            algorithm=args.algorithm,
            dataset=args.dataset,
            n_iterations=args.iterations,
            n_epochs_per_eval=args.epochs,
            seed=args.seed,
        )
        result = tuner.run()
        if args.plot:
            from cnn_hypertuner.utils.plotting import plot_convergence
            plot_convergence(result, save_path=f"experiments/{result.experiment_name}_convergence.png")


if __name__ == "__main__":
    main()
