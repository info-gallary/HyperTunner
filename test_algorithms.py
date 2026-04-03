"""
Unit tests for CNN HyperTuner.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from cnn_hypertuner.search_space import SearchSpace, HyperparameterSpace
from cnn_hypertuner.algorithms import GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing


# ─── Mock objective function ──────────────────────────────────────────────────

def mock_objective(config: dict) -> float:
    """Sphere function mapped to [0,1] — fast, no GPU needed."""
    lr = float(config.get("learning_rate", 0.001))
    dr = float(config.get("dropout_rate", 0.3))
    score = 1.0 - ((np.log10(lr) + 4) / 4) ** 2 - dr ** 2
    return float(np.clip(score, 0.0, 1.0))


# ─── SearchSpace Tests ────────────────────────────────────────────────────────

class TestSearchSpace:
    def setup_method(self):
        self.space = SearchSpace()

    def test_sample_random_has_all_keys(self):
        config = self.space.sample_random()
        for name in self.space.param_names:
            assert name in config

    def test_vector_round_trip(self):
        """Round-trip: config → vector → config should reproduce same values."""
        config = self.space.sample_random()
        vec = self.space.config_to_vector(config)
        config2 = self.space.vector_to_config(vec)
        for k in config:
            hp = self.space.params[k]
            if hp.hp_type == "categorical":
                assert config[k] == config2[k]
            elif hp.hp_type == "int":
                assert int(config[k]) == int(config2[k])
            else:
                # float/log-scale: compare in original space with relative tolerance
                v1, v2 = float(config[k]), float(config2[k])
                assert abs(v1 - v2) / (abs(v1) + 1e-12) < 0.05

    def test_bounds_shape(self):
        lows, highs = self.space.get_bounds()
        assert len(lows) == self.space.dim
        assert len(highs) == self.space.dim
        assert all(lows <= highs)

    def test_clip_stays_in_bounds(self):
        lows, highs = self.space.get_bounds()
        wild = lows - 999
        clipped = self.space.clip_vector(wild)
        l2, h2 = self.space.get_bounds()
        assert all(clipped >= l2 - 1e-9)

    def test_custom_space(self):
        custom = {
            "lr": HyperparameterSpace("lr", "float", 1e-4, 1e-2, log_scale=True),
            "bs": HyperparameterSpace("bs", "categorical", choices=[32, 64]),
        }
        space = SearchSpace(custom)
        # custom params are merged with defaults; check they exist
        assert "lr" in space.params
        assert "bs" in space.params
        config = space.sample_random()
        assert config["bs"] in [32, 64]


# ─── Algorithm Tests ──────────────────────────────────────────────────────────

class TestGeneticAlgorithm:
    def setup_method(self):
        self.space = SearchSpace()

    def test_ga_runs_and_returns_config(self):
        ga = GeneticAlgorithm(
            self.space, mock_objective,
            population_size=5, seed=0, verbose=False
        )
        best = ga.optimize(n_iterations=3)
        assert isinstance(best, dict)
        for k in self.space.param_names:
            assert k in best

    def test_ga_improves(self):
        ga = GeneticAlgorithm(
            self.space, mock_objective,
            population_size=8, seed=1, verbose=False
        )
        ga.optimize(n_iterations=5)
        curve = ga.get_convergence_curve()
        assert curve[-1] >= curve[0]  # monotonically non-decreasing

    def test_ga_summary(self):
        ga = GeneticAlgorithm(
            self.space, mock_objective,
            population_size=5, seed=2, verbose=False
        )
        ga.optimize(n_iterations=2)
        s = ga.summary()
        assert "best_score" in s
        assert "best_config" in s
        assert s["total_evaluations"] > 0


class TestPSO:
    def setup_method(self):
        self.space = SearchSpace()

    def test_pso_global_topology(self):
        pso = ParticleSwarmOptimization(
            self.space, mock_objective,
            n_particles=5, seed=0, verbose=False, topology="global"
        )
        best = pso.optimize(n_iterations=3)
        assert isinstance(best, dict)

    def test_pso_ring_topology(self):
        pso = ParticleSwarmOptimization(
            self.space, mock_objective,
            n_particles=5, seed=0, verbose=False, topology="ring"
        )
        best = pso.optimize(n_iterations=3)
        assert isinstance(best, dict)

    def test_pso_score_non_negative(self):
        pso = ParticleSwarmOptimization(
            self.space, mock_objective,
            n_particles=5, seed=3, verbose=False
        )
        pso.optimize(n_iterations=3)
        assert pso.best_score >= 0.0


class TestSimulatedAnnealing:
    def setup_method(self):
        self.space = SearchSpace()

    def test_sa_gaussian(self):
        sa = SimulatedAnnealing(
            self.space, mock_objective,
            T0=1.0, alpha=0.9, seed=0, verbose=False,
            perturbation="gaussian"
        )
        best = sa.optimize(n_iterations=10)
        assert isinstance(best, dict)

    def test_sa_cauchy(self):
        sa = SimulatedAnnealing(
            self.space, mock_objective,
            T0=1.0, alpha=0.9, seed=0, verbose=False,
            perturbation="cauchy"
        )
        best = sa.optimize(n_iterations=10)
        assert isinstance(best, dict)

    def test_sa_acceptance_log(self):
        sa = SimulatedAnnealing(
            self.space, mock_objective,
            T0=1.0, alpha=0.8, seed=42, verbose=False
        )
        sa.optimize(n_iterations=10)
        assert len(sa.acceptance_log) > 0
        assert all(0.0 <= r <= 1.0 for r in sa.acceptance_log)


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_all_algorithms_produce_valid_configs(self):
        space = SearchSpace()
        for AlgoClass in [GeneticAlgorithm, ParticleSwarmOptimization, SimulatedAnnealing]:
            optimizer = AlgoClass(space, mock_objective, seed=99, verbose=False,
                                  **{"population_size": 4} if AlgoClass == GeneticAlgorithm else
                                  {"n_particles": 4} if AlgoClass == ParticleSwarmOptimization else {})
            best = optimizer.optimize(n_iterations=3)
            assert isinstance(best, dict)
            assert optimizer.best_score >= 0.0
