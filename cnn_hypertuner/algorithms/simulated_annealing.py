"""
Simulated Annealing (SA) for CNN Hyperparameter Optimization.

Features:
  - Geometric cooling schedule (T *= alpha each step)
  - Cauchy/Gaussian neighbor generation (switchable)
  - Reheating on stagnation
  - Tracks acceptance rate for diagnostics
"""

import numpy as np
import logging
from typing import Callable, Dict, Any, List, Optional, Tuple
import time
import math

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class SimulatedAnnealing(BaseOptimizer):
    """
    Simulated Annealing optimizer with:
      - Geometric cooling: T(k) = T0 * alpha^k
      - Metropolis acceptance criterion
      - Cauchy or Gaussian perturbation
      - Automatic reheat on stagnation
      - Best solution tracking separate from current
    """

    def __init__(
        self,
        search_space,
        objective_fn: Callable[[Dict], float],
        T0: float = 1.0,          # Initial temperature
        T_min: float = 1e-5,      # Minimum temperature
        alpha: float = 0.97,      # Cooling rate
        perturbation: str = "gaussian",   # "gaussian" or "cauchy"
        perturbation_scale: float = 0.1,  # Fraction of range
        reheat_factor: float = 2.0,
        stagnation_reheat_after: int = 30,
        maximize: bool = True,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_fn: Optional[Callable] = None,
    ):
        super().__init__(search_space, objective_fn, maximize, seed, verbose, checkpoint_fn)
        self.T0 = T0
        self.T_min = T_min
        self.alpha = alpha
        self.perturbation = perturbation
        self.perturbation_scale = perturbation_scale
        self.reheat_factor = reheat_factor
        self.stagnation_reheat_after = stagnation_reheat_after

        self.temperature_log: List[float] = []
        self.acceptance_log: List[float] = []

    def _perturb(self, vector: np.ndarray, temperature: float) -> np.ndarray:
        """Generate neighbor solution by perturbing one or more dimensions."""
        lows, highs = self.space.get_bounds()
        new_vec = vector.copy()
        # Perturb a random number of dimensions (1 to dim//3)
        n_perturb = np.random.randint(1, max(2, self.space.dim // 3))
        dims = np.random.choice(self.space.dim, n_perturb, replace=False)

        for d in dims:
            scale = self.perturbation_scale * (highs[d] - lows[d]) * (temperature / self.T0)
            if self.perturbation == "cauchy":
                delta = np.random.standard_cauchy() * scale
            else:
                delta = np.random.normal(0, scale)
            new_vec[d] = np.clip(new_vec[d] + delta, lows[d], highs[d])
        return new_vec

    def _accept(self, current_score: float, new_score: float, T: float) -> Tuple[bool, float]:
        """Metropolis criterion."""
        if self.maximize:
            delta = new_score - current_score
        else:
            delta = current_score - new_score

        if delta > 0:
            return True, 1.0
        prob = math.exp(delta / max(T, 1e-300))
        return np.random.rand() < prob, prob

    def optimize(self, n_iterations: int = 500, **kwargs) -> Dict[str, Any]:
        self._start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"  Simulated Annealing — Iters={n_iterations} T0={self.T0} α={self.alpha}")
        logger.info(f"  Perturbation={self.perturbation} | TMin={self.T_min:.2e}")
        logger.info(f"{'='*60}")

        # Start from random solution
        current_vec = self.space.random_vector()
        current_config = self.space.vector_to_config(current_vec)
        current_score = self._evaluate(current_config, iteration=0)

        T = self.T0
        stagnation = 0
        prev_best = current_score
        accepted_count = 0

        for it in range(1, n_iterations + 1):
            if T < self.T_min:
                if self.verbose:
                    logger.info(f"  → T below T_min at iter {it}. Stopping.")
                break

            # Generate and evaluate neighbor
            new_vec = self._perturb(current_vec, T)
            new_config = self.space.vector_to_config(new_vec)
            new_score = self._evaluate(new_config, iteration=it)

            # Accept / reject
            accepted, prob = self._accept(current_score, new_score, T)
            if accepted:
                current_vec = new_vec
                current_score = new_score
                accepted_count += 1

            # Cooling
            T *= self.alpha
            self.temperature_log.append(T)

            # Stagnation detection
            if self.best_score == prev_best:
                stagnation += 1
            else:
                stagnation = 0
                prev_best = self.best_score

            if stagnation >= self.stagnation_reheat_after:
                T = min(T * self.reheat_factor, self.T0 * 0.5)
                stagnation = 0
                if self.verbose:
                    logger.info(f"  ↑ Reheat at iter {it}. T={T:.4f}")

            # Logging
            if self.verbose and it % max(1, n_iterations // 20) == 0:
                acc_rate = accepted_count / it
                logger.info(
                    f"  Iter {it:4d}/{n_iterations} | T={T:.4f} "
                    f"cur={current_score:.4f} best={self.best_score:.4f} "
                    f"acc={acc_rate:.2f}"
                )
            self.acceptance_log.append(accepted_count / it)

        logger.info(f"\n  ✓ SA complete. Best score: {self.best_score:.4f}")
        logger.info(f"  Best config: {self.best_config}")
        return self.best_config
