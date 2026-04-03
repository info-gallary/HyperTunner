"""
CNNHyperTuner — Top-level orchestrator.
Ties together SearchSpace, Trainer, and chosen Algorithm.
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from .search_space import SearchSpace
from .trainer import CNNTrainer
from .algorithms import GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing

logger = logging.getLogger(__name__)


ALGORITHM_MAP = {
    "ga":  GeneticAlgorithm,
    "pso": ParticleSwarmOptimization,
    "sa":  SimulatedAnnealing,
    "genetic_algorithm":       GeneticAlgorithm,
    "particle_swarm":          ParticleSwarmOptimization,
    "simulated_annealing":     SimulatedAnnealing,
}


class CNNHyperTuner:
    """
    Main entry point for CNN hyperparameter tuning using soft computing.

    Example usage:
        tuner = CNNHyperTuner(algorithm="ga", dataset="cifar10", n_iterations=30)
        result = tuner.run()
        print(result.best_config)
    """

    def __init__(
        self,
        algorithm: str = "pso",
        dataset: str = "cifar10",
        n_iterations: int = 30,
        n_epochs_per_eval: int = 5,
        output_dir: str = "./experiments",
        experiment_name: Optional[str] = None,
        device: Optional[str] = None,
        seed: int = 42,
        verbose: bool = True,
        custom_search_space: Optional[SearchSpace] = None,
        custom_train_loader=None,
        custom_val_loader=None,
        algorithm_kwargs: Optional[Dict] = None,
    ):
        self.algorithm_name = algorithm.lower()
        self.dataset = dataset
        self.n_iterations = n_iterations
        self.n_epochs = n_epochs_per_eval
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or f"{algorithm}_{dataset}_{int(time.time())}"
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.algorithm_kwargs = algorithm_kwargs or {}

        self.search_space = custom_search_space or SearchSpace()
        self._setup_logging()

        # Build trainer
        if custom_train_loader:
            self.trainer = CNNTrainer(
                dataset_name="custom",
                n_epochs=n_epochs_per_eval,
                device=device,
                custom_train_loader=custom_train_loader,
                custom_val_loader=custom_val_loader,
            )
        else:
            self.trainer = CNNTrainer(
                dataset_name=dataset,
                n_epochs=n_epochs_per_eval,
                device=device,
            )

        self.optimizer = None
        self.result = None

    def _setup_logging(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.output_dir / f"{self.experiment_name}.log"
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )

    def _make_checkpoint_fn(self):
        checkpoint_path = self.output_dir / f"{self.experiment_name}_checkpoint.json"

        def checkpoint_fn(result, best_config, best_score):
            data = {
                "best_score": best_score,
                "best_config": best_config,
                "n_evaluations": result.iteration,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(data, f, indent=2)

        return checkpoint_fn

    def run(self) -> "TunerResult":
        logger.info(f"\n{'#'*60}")
        logger.info(f"  CNN HyperTuner — Experiment: {self.experiment_name}")
        logger.info(f"  Algorithm: {self.algorithm_name.upper()} | Dataset: {self.dataset}")
        logger.info(f"  Iterations: {self.n_iterations} | Epochs/eval: {self.n_epochs}")
        logger.info(f"{'#'*60}")
        logger.info(str(self.search_space))

        if self.algorithm_name not in ALGORITHM_MAP:
            raise ValueError(f"Unknown algorithm '{self.algorithm_name}'. Choose from: {list(ALGORITHM_MAP.keys())}")

        AlgoClass = ALGORITHM_MAP[self.algorithm_name]
        checkpoint_fn = self._make_checkpoint_fn()

        self.optimizer = AlgoClass(
            search_space=self.search_space,
            objective_fn=self.trainer.get_objective_fn(),
            maximize=True,
            seed=self.seed,
            verbose=self.verbose,
            checkpoint_fn=checkpoint_fn,
            **self.algorithm_kwargs,
        )

        start = time.time()
        best_config = self.optimizer.optimize(n_iterations=self.n_iterations)
        elapsed = time.time() - start

        summary = self.optimizer.summary()
        self.result = TunerResult(
            best_config=best_config,
            best_score=summary["best_score"],
            algorithm=self.algorithm_name,
            total_evaluations=summary["total_evaluations"],
            elapsed_sec=elapsed,
            convergence_curve=summary["convergence_curve"],
            history=summary["history"],
            experiment_name=self.experiment_name,
        )

        self._save_results()
        self.result.print_summary()
        return self.result

    def _save_results(self):
        out = self.output_dir / f"{self.experiment_name}_results.json"
        with open(out, "w") as f:
            json.dump({
                "experiment": self.experiment_name,
                "algorithm": self.algorithm_name,
                "dataset": self.dataset,
                "best_score": self.result.best_score,
                "best_config": self.result.best_config,
                "total_evaluations": self.result.total_evaluations,
                "elapsed_sec": self.result.elapsed_sec,
                "convergence_curve": self.result.convergence_curve,
            }, f, indent=2)
        logger.info(f"\n  Results saved to: {out}")

    @staticmethod
    def compare_algorithms(
        algorithms: List[str],
        dataset: str = "cifar10",
        n_iterations: int = 20,
        n_epochs_per_eval: int = 3,
        seed: int = 42,
        output_dir: str = "./experiments/comparison",
    ) -> Dict[str, "TunerResult"]:
        """Run multiple algorithms and compare results."""
        results = {}
        for algo in algorithms:
            tuner = CNNHyperTuner(
                algorithm=algo,
                dataset=dataset,
                n_iterations=n_iterations,
                n_epochs_per_eval=n_epochs_per_eval,
                output_dir=output_dir,
                seed=seed,
            )
            results[algo] = tuner.run()

        # Print comparison table
        print("\n" + "="*60)
        print("  Algorithm Comparison")
        print("="*60)
        print(f"  {'Algorithm':15s} {'Best Score':>12s} {'Evaluations':>12s} {'Time (s)':>10s}")
        print("-"*60)
        for algo, r in results.items():
            print(f"  {algo:15s} {r.best_score:>12.4f} {r.total_evaluations:>12d} {r.elapsed_sec:>10.1f}")
        print("="*60)
        return results


class TunerResult:
    """Encapsulates the output of a tuning run."""

    def __init__(self, best_config, best_score, algorithm, total_evaluations,
                 elapsed_sec, convergence_curve, history, experiment_name):
        self.best_config = best_config
        self.best_score = best_score
        self.algorithm = algorithm
        self.total_evaluations = total_evaluations
        self.elapsed_sec = elapsed_sec
        self.convergence_curve = convergence_curve
        self.history = history
        self.experiment_name = experiment_name

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  ✓ Tuning Complete — {self.experiment_name}")
        print(f"{'='*60}")
        print(f"  Algorithm     : {self.algorithm.upper()}")
        print(f"  Best Score    : {self.best_score:.4f}")
        print(f"  Total Evals   : {self.total_evaluations}")
        print(f"  Time Elapsed  : {self.elapsed_sec:.1f}s")
        print(f"\n  Best Hyperparameters:")
        for k, v in self.best_config.items():
            print(f"    {k:25s} = {v}")
        print("="*60)

    def __repr__(self):
        return f"TunerResult(score={self.best_score:.4f}, algo={self.algorithm}, evals={self.total_evaluations})"
