"""
Genetic Algorithm (GA) for CNN Hyperparameter Optimization.

Strategy:
  - Real-valued chromosome encoding of the search space
  - Tournament selection
  - Blend crossover (BLX-α)
  - Gaussian mutation with adaptive sigma
  - Elitism preservation
"""

import numpy as np
import logging
from typing import Callable, Dict, Any, List, Optional, Tuple
import time

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class Individual:
    def __init__(self, vector: np.ndarray, score: float = -np.inf):
        self.vector = vector.copy()
        self.score = score

    def copy(self):
        return Individual(self.vector.copy(), self.score)


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm optimizer with:
      - Tournament selection
      - BLX-α crossover
      - Adaptive Gaussian mutation
      - Elitism
      - Optional restart on stagnation
    """

    def __init__(
        self,
        search_space,
        objective_fn: Callable[[Dict], float],
        population_size: int = 20,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        elitism_count: int = 2,
        blx_alpha: float = 0.5,
        mutation_sigma: float = 0.1,
        adaptive_mutation: bool = True,
        stagnation_restart_after: int = 20,
        maximize: bool = True,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_fn: Optional[Callable] = None,
    ):
        super().__init__(search_space, objective_fn, maximize, seed, verbose, checkpoint_fn)
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.blx_alpha = blx_alpha
        self.mutation_sigma = mutation_sigma
        self.adaptive_mutation = adaptive_mutation
        self.stagnation_restart_after = stagnation_restart_after

        self.population: List[Individual] = []
        self.generation_stats: List[Dict] = []

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize_population(self) -> List[Individual]:
        return [Individual(self.space.random_vector()) for _ in range(self.pop_size)]

    # ─── Evaluation ───────────────────────────────────────────────────────────

    def _evaluate_population(self, population: List[Individual], generation: int) -> None:
        for ind in population:
            if ind.score == -np.inf:
                config = self.space.vector_to_config(ind.vector)
                ind.score = self._evaluate(config, generation)

    # ─── Selection ────────────────────────────────────────────────────────────

    def _tournament_select(self, population: List[Individual]) -> Individual:
        contestants = np.random.choice(len(population), self.tournament_size, replace=False)
        best = max(contestants, key=lambda i: population[i].score if self.maximize else -population[i].score)
        return population[best].copy()

    # ─── Crossover ────────────────────────────────────────────────────────────

    def _blx_alpha_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        """Blend crossover (BLX-α): explores beyond parent range."""
        lows, highs = self.space.get_bounds()
        c1_vec = np.zeros_like(p1.vector)
        c2_vec = np.zeros_like(p2.vector)

        for i in range(len(p1.vector)):
            lo = min(p1.vector[i], p2.vector[i])
            hi = max(p1.vector[i], p2.vector[i])
            interval = hi - lo
            lo_blend = max(lows[i], lo - self.blx_alpha * interval)
            hi_blend = min(highs[i], hi + self.blx_alpha * interval)
            c1_vec[i] = np.random.uniform(lo_blend, hi_blend)
            c2_vec[i] = np.random.uniform(lo_blend, hi_blend)

        return Individual(c1_vec), Individual(c2_vec)

    # ─── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, ind: Individual, sigma: float) -> Individual:
        lows, highs = self.space.get_bounds()
        vector = ind.vector.copy()
        for i in range(len(vector)):
            if np.random.rand() < self.mutation_rate:
                noise = np.random.normal(0, sigma * (highs[i] - lows[i]))
                vector[i] = np.clip(vector[i] + noise, lows[i], highs[i])
        return Individual(vector)

    # ─── Main Loop ────────────────────────────────────────────────────────────

    def optimize(self, n_iterations: int = 50, **kwargs) -> Dict[str, Any]:
        self._start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"  Genetic Algorithm — Pop={self.pop_size}, Gens={n_iterations}")
        logger.info(f"{'='*60}")

        self.population = self._initialize_population()
        self._evaluate_population(self.population, generation=0)

        stagnation_counter = 0
        sigma = self.mutation_sigma
        prev_best = -np.inf if self.maximize else np.inf

        for gen in range(1, n_iterations + 1):
            # Sort population
            self.population.sort(key=lambda x: x.score, reverse=self.maximize)
            gen_best = self.population[0].score

            # Stagnation detection + adaptive mutation
            if gen_best == prev_best:
                stagnation_counter += 1
                if self.adaptive_mutation:
                    sigma = min(0.5, sigma * 1.05)
            else:
                stagnation_counter = 0
                if self.adaptive_mutation:
                    sigma = max(0.01, sigma * 0.98)

            # Restart on stagnation
            if stagnation_counter >= self.stagnation_restart_after:
                if self.verbose:
                    logger.info(f"  ↻ Stagnation detected at gen {gen}. Partial restart.")
                elites = self.population[:self.elitism_count]
                restarts = [Individual(self.space.random_vector())
                            for _ in range(self.pop_size - self.elitism_count)]
                self.population = elites + restarts
                self._evaluate_population(self.population, gen)
                stagnation_counter = 0
                sigma = self.mutation_sigma
                continue

            prev_best = gen_best

            # Elitism
            elites = [ind.copy() for ind in self.population[:self.elitism_count]]

            # Generate offspring
            offspring = []
            while len(offspring) < self.pop_size - self.elitism_count:
                p1 = self._tournament_select(self.population)
                p2 = self._tournament_select(self.population)
                if np.random.rand() < self.crossover_rate:
                    c1, c2 = self._blx_alpha_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                offspring.extend([self._mutate(c1, sigma), self._mutate(c2, sigma)])

            self.population = elites + offspring[:self.pop_size - self.elitism_count]
            self._evaluate_population(self.population, gen)

            # Stats
            scores = [ind.score for ind in self.population]
            stat = {
                "generation": gen,
                "best": max(scores) if self.maximize else min(scores),
                "mean": np.mean(scores),
                "std": np.std(scores),
                "sigma": sigma,
            }
            self.generation_stats.append(stat)

            if self.verbose and gen % 5 == 0:
                logger.info(
                    f"  Gen {gen:3d}/{n_iterations} | best={stat['best']:.4f} "
                    f"mean={stat['mean']:.4f} std={stat['std']:.4f} σ={sigma:.3f}"
                )

        logger.info(f"\n  ✓ GA complete. Best score: {self.best_score:.4f}")
        logger.info(f"  Best config: {self.best_config}")
        return self.best_config
