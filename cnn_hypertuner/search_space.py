"""
Search Space — Defines the hyperparameter bounds for CNN tuning.
All algorithms operate within this unified space.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class HyperparameterSpace:
    """Single hyperparameter definition."""
    name: str
    hp_type: str          # 'int', 'float', 'categorical'
    low: float = 0.0
    high: float = 1.0
    choices: List[Any] = field(default_factory=list)
    log_scale: bool = False

    def sample(self) -> Any:
        if self.hp_type == "categorical":
            return np.random.choice(self.choices)
        elif self.hp_type == "int":
            return int(np.random.randint(int(self.low), int(self.high) + 1))
        elif self.hp_type == "float":
            if self.log_scale:
                return float(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))))
            return float(np.random.uniform(self.low, self.high))

    def clip(self, value: float) -> Any:
        if self.hp_type == "categorical":
            idx = int(np.clip(round(value), 0, len(self.choices) - 1))
            return self.choices[idx]
        elif self.hp_type == "int":
            return int(np.clip(round(value), self.low, self.high))
        elif self.hp_type == "float":
            val = float(np.clip(value, self.low, self.high))
            if self.log_scale:
                val = float(np.clip(np.exp(value), self.low, self.high))
            return val

    def to_numeric(self, value: Any) -> float:
        if self.hp_type == "categorical":
            return float(self.choices.index(value))
        if self.log_scale:
            return float(np.log(max(value, 1e-300)))
        return float(value)

    def from_numeric(self, value: float) -> Any:
        return self.clip(value)

    @property
    def numeric_bounds(self) -> Tuple[float, float]:
        if self.hp_type == "categorical":
            return (0.0, float(len(self.choices) - 1))
        if self.log_scale:
            return (np.log(self.low), np.log(self.high))
        return (float(self.low), float(self.high))


class SearchSpace:
    """
    Unified search space for CNN hyperparameters.
    Configurable — override any defaults by passing a custom dict.
    """

    DEFAULT_SPACE = {
        "learning_rate": HyperparameterSpace("learning_rate", "float", 1e-5, 1e-1, log_scale=True),
        "batch_size":    HyperparameterSpace("batch_size", "categorical", choices=[16, 32, 64, 128, 256]),
        "num_filters_1": HyperparameterSpace("num_filters_1", "int", 16, 128),
        "num_filters_2": HyperparameterSpace("num_filters_2", "int", 32, 256),
        "num_filters_3": HyperparameterSpace("num_filters_3", "int", 64, 512),
        "kernel_size":   HyperparameterSpace("kernel_size", "categorical", choices=[3, 5, 7]),
        "dropout_rate":  HyperparameterSpace("dropout_rate", "float", 0.0, 0.7),
        "dense_units":   HyperparameterSpace("dense_units", "int", 64, 1024),
        "optimizer":     HyperparameterSpace("optimizer", "categorical", choices=["adam", "sgd", "rmsprop", "adamw"]),
        "weight_decay":  HyperparameterSpace("weight_decay", "float", 1e-6, 1e-2, log_scale=True),
        "activation":    HyperparameterSpace("activation", "categorical", choices=["relu", "leaky_relu", "elu", "gelu"]),
        "num_conv_layers": HyperparameterSpace("num_conv_layers", "int", 2, 5),
    }

    def __init__(self, custom_space: Dict[str, HyperparameterSpace] = None):
        self.params = {**self.DEFAULT_SPACE, **(custom_space or {})}
        self.param_names = list(self.params.keys())
        self.dim = len(self.param_names)

    def sample_random(self) -> Dict[str, Any]:
        return {name: hp.sample() for name, hp in self.params.items()}

    def vector_to_config(self, vector: np.ndarray) -> Dict[str, Any]:
        config = {}
        for i, name in enumerate(self.param_names):
            config[name] = self.params[name].from_numeric(vector[i])
        return config

    def config_to_vector(self, config: Dict[str, Any]) -> np.ndarray:
        return np.array([self.params[name].to_numeric(config[name])
                         for name in self.param_names], dtype=float)

    def random_vector(self) -> np.ndarray:
        return self.config_to_vector(self.sample_random())

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lows, highs = [], []
        for name in self.param_names:
            lo, hi = self.params[name].numeric_bounds
            lows.append(lo)
            highs.append(hi)
        return np.array(lows), np.array(highs)

    def clip_vector(self, vector: np.ndarray) -> np.ndarray:
        lows, highs = self.get_bounds()
        return np.clip(vector, lows, highs)

    def __repr__(self):
        lines = [f"SearchSpace ({self.dim} params):"]
        for name, hp in self.params.items():
            if hp.hp_type == "categorical":
                lines.append(f"  {name:20s} categorical {hp.choices}")
            elif hp.hp_type == "int":
                lines.append(f"  {name:20s} int         [{int(hp.low)}, {int(hp.high)}]")
            else:
                lines.append(f"  {name:20s} float       [{hp.low:.2e}, {hp.high:.2e}]{'  (log)' if hp.log_scale else ''}")
        return "\n".join(lines)
