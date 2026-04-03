"""
Particle Swarm Optimization (PSO) for CNN Hyperparameter Optimization.

Features:
  - Inertia weight decay
  - Cognitive + social acceleration
  - Velocity clamping
  - Local best topology (ring) for diversity
  - Restart on stagnation
"""

import numpy as np
import logging
from typing import Callable, Dict, Any, List, Optional
import time

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class Particle:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.score: float = -np.inf
        self.best_position = position.copy()
        self.best_score: float = -np.inf

    def update_personal_best(self, maximize: bool = True):
        is_better = self.score > self.best_score if maximize else self.score < self.best_score
        if is_better:
            self.best_score = self.score
            self.best_position = self.position.copy()


class ParticleSwarmOptimization(BaseOptimizer):
    """
    PSO with:
      - Linear inertia weight decay (w_max → w_min)
      - Cognitive (c1) and social (c2) components
      - Velocity clamping (v_max = 20% of range)
      - Ring topology for local best (improves diversity)
      - Stagnation restart
    """

    def __init__(
        self,
        search_space,
        objective_fn: Callable[[Dict], float],
        n_particles: int = 25,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,      # cognitive (personal best)
        c2: float = 2.0,      # social (global best)
        v_max_fraction: float = 0.2,
        topology: str = "global",  # "global" or "ring"
        ring_neighbors: int = 2,
        stagnation_restart_after: int = 20,
        maximize: bool = True,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_fn: Optional[Callable] = None,
    ):
        super().__init__(search_space, objective_fn, maximize, seed, verbose, checkpoint_fn)
        self.n_particles = n_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.v_max_fraction = v_max_fraction
        self.topology = topology
        self.ring_neighbors = ring_neighbors
        self.stagnation_restart_after = stagnation_restart_after

        self.swarm: List[Particle] = []
        self.iteration_stats: List[Dict] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_score: float = -np.inf if maximize else np.inf

    def _initialize_swarm(self) -> List[Particle]:
        lows, highs = self.space.get_bounds()
        particles = []
        for _ in range(self.n_particles):
            pos = np.array([np.random.uniform(lo, hi) for lo, hi in zip(lows, highs)])
            vel_range = (highs - lows) * self.v_max_fraction
            vel = np.array([np.random.uniform(-r, r) for r in vel_range])
            particles.append(Particle(pos, vel))
        return particles

    def _get_v_max(self) -> np.ndarray:
        lows, highs = self.space.get_bounds()
        return (highs - lows) * self.v_max_fraction

    def _get_local_best(self, idx: int) -> np.ndarray:
        """Ring topology: best among neighbors."""
        n = len(self.swarm)
        neighbors = [(idx + k) % n for k in range(-self.ring_neighbors, self.ring_neighbors + 1)]
        if self.maximize:
            best_neighbor = max(neighbors, key=lambda i: self.swarm[i].best_score)
        else:
            best_neighbor = min(neighbors, key=lambda i: self.swarm[i].best_score)
        return self.swarm[best_neighbor].best_position

    def _update_global_best(self):
        for p in self.swarm:
            is_better = p.best_score > self.global_best_score if self.maximize else p.best_score < self.global_best_score
            if is_better:
                self.global_best_score = p.best_score
                self.global_best_position = p.best_position.copy()

    def optimize(self, n_iterations: int = 50, **kwargs) -> Dict[str, Any]:
        self._start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"  Particle Swarm Optimization — N={self.n_particles}, Iters={n_iterations}")
        logger.info(f"  Topology={self.topology} | c1={self.c1} c2={self.c2}")
        logger.info(f"{'='*60}")

        lows, highs = self.space.get_bounds()
        v_max = self._get_v_max()

        self.swarm = self._initialize_swarm()

        # Initial evaluation
        for p in self.swarm:
            config = self.space.vector_to_config(p.position)
            p.score = self._evaluate(config, iteration=0)
            p.update_personal_best(self.maximize)
        self._update_global_best()

        stagnation = 0
        prev_best = self.global_best_score

        for it in range(1, n_iterations + 1):
            # Linearly decay inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (it / n_iterations)

            for i, p in enumerate(self.swarm):
                r1 = np.random.rand(self.space.dim)
                r2 = np.random.rand(self.space.dim)

                if self.topology == "ring":
                    attractor = self._get_local_best(i)
                else:
                    attractor = self.global_best_position

                # Velocity update
                cognitive = self.c1 * r1 * (p.best_position - p.position)
                social    = self.c2 * r2 * (attractor - p.position)
                p.velocity = w * p.velocity + cognitive + social

                # Clamp velocity
                p.velocity = np.clip(p.velocity, -v_max, v_max)

                # Position update
                p.position = np.clip(p.position + p.velocity, lows, highs)

                # Evaluate
                config = self.space.vector_to_config(p.position)
                p.score = self._evaluate(config, iteration=it)
                p.update_personal_best(self.maximize)

            self._update_global_best()

            # Stagnation check
            if self.global_best_score == prev_best:
                stagnation += 1
            else:
                stagnation = 0
            prev_best = self.global_best_score

            if stagnation >= self.stagnation_restart_after:
                if self.verbose:
                    logger.info(f"  ↻ Stagnation at iter {it}. Re-scattering swarm.")
                # Keep best particle, restart rest
                best_particle = max(self.swarm, key=lambda p: p.best_score if self.maximize else -p.best_score)
                self.swarm = self._initialize_swarm()
                self.swarm[0] = best_particle
                for p in self.swarm[1:]:
                    config = self.space.vector_to_config(p.position)
                    p.score = self._evaluate(config, iteration=it)
                    p.update_personal_best(self.maximize)
                self._update_global_best()
                stagnation = 0
                continue

            # Stats
            scores = [p.score for p in self.swarm]
            stat = {
                "iteration": it,
                "global_best": self.global_best_score,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "w": w,
            }
            self.iteration_stats.append(stat)

            if self.verbose and it % 5 == 0:
                logger.info(
                    f"  Iter {it:3d}/{n_iterations} | best={stat['global_best']:.4f} "
                    f"mean={stat['mean']:.4f} w={w:.3f}"
                )

        self.best_config = self.space.vector_to_config(self.global_best_position)
        self.best_score = self.global_best_score
        logger.info(f"\n  ✓ PSO complete. Best score: {self.best_score:.4f}")
        logger.info(f"  Best config: {self.best_config}")
        return self.best_config
