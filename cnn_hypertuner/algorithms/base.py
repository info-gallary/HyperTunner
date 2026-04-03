import numpy as np
from typing import Callable, Dict, Any, Optional

class BaseOptimizer:
    def __init__(
        self,
        search_space,
        objective_fn: Callable[[Dict], float],
        maximize: bool = True,
        seed: int = 42,
        verbose: bool = True,
        checkpoint_fn: Optional[Callable] = None,
    ):
        self.space = search_space
        self.objective_fn = objective_fn
        self.maximize = maximize
        self.seed = seed
        self.verbose = verbose
        self.checkpoint_fn = checkpoint_fn

        self.best_config = None
        self.best_score = -np.inf if maximize else np.inf
        self.total_evaluations = 0
        self.convergence_curve = []
        self.history = []
        self._start_time = None
        # for SA since SA tracks iteration using something else maybe? No.
        self.iteration = 0 

        np.random.seed(seed)

    def _evaluate(self, config: Dict, iteration: int) -> float:
        self.iteration = iteration
        score = self.objective_fn(config)
        self.total_evaluations += 1

        is_best = False
        if self.best_config is None:
            is_best = True
        elif self.maximize and score > self.best_score:
            is_best = True
        elif not self.maximize and score < self.best_score:
            is_best = True

        if is_best:
            self.best_score = score
            self.best_config = config
            if self.checkpoint_fn:
                self.checkpoint_fn(self, self.best_config, self.best_score)

        stat = {"iteration": iteration, "score": score, "config": config}
        self.history.append(stat)
        self.convergence_curve.append(self.best_score)
        return score

    def summary(self) -> Dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_config": self.best_config,
            "total_evaluations": self.total_evaluations,
            "convergence_curve": self.convergence_curve,
            "history": self.history
        }
