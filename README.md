# 🧬 CNN HyperTuner

> Soft Computing Hyperparameter Optimization for Convolutional Neural Networks  
> **GA · PSO · SA** — all three algorithms, switchable in one command.

---

## Features

| Feature | Detail |
|---|---|
| 🧬 **Genetic Algorithm** | BLX-α crossover, tournament selection, adaptive mutation, elitism |
| 🐦 **Particle Swarm** | Inertia decay, ring/global topology, velocity clamping, stagnation restart |
| 🌡 **Simulated Annealing** | Cauchy/Gaussian perturbation, reheating, Metropolis criterion |
| 🧠 **FlexCNN** | Depth 2–5 layers, variable filters, activations, optimizers — all tunable |
| 📊 **Visualization** | Convergence curves, algorithm comparison plots |
| 💾 **Checkpointing** | Auto-saves best config every evaluation |
| 🔌 **Custom Dataset** | Plug in your own DataLoader |

---

## Installation

```bash
cd cnn_hypertuner
pip install -e .
# With visualization:
pip install -e ".[viz]"
```

---

## Quick Start

### Python API

```python
from cnn_hypertuner import CNNHyperTuner

# Tune with PSO on CIFAR-10
tuner = CNNHyperTuner(
    algorithm="pso",
    dataset="cifar10",
    n_iterations=30,
    n_epochs_per_eval=5,
)
result = tuner.run()

print(result.best_config)
print(f"Best Accuracy: {result.best_score:.4f}")
```

### Switch Algorithms

```python
# Genetic Algorithm
tuner = CNNHyperTuner(algorithm="ga", dataset="mnist", n_iterations=40)
result = tuner.run()

# Simulated Annealing
tuner = CNNHyperTuner(algorithm="sa", dataset="cifar10", n_iterations=200)
result = tuner.run()
```

### Compare All Three

```python
results = CNNHyperTuner.compare_algorithms(
    algorithms=["ga", "pso", "sa"],
    dataset="cifar10",
    n_iterations=20,
    n_epochs_per_eval=3,
)
```

### CLI

```bash
# Tune
cnn-hypertuner tune --algorithm pso --dataset cifar10 --iterations 30

# Compare
cnn-hypertuner compare --algorithms ga pso sa

# Info
cnn-hypertuner info
```

### Script

```bash
python run_tuner.py --algorithm ga --dataset cifar10 --iterations 30 --plot
python run_tuner.py --compare --plot
```

---

## Custom Dataset

```python
from torch.utils.data import DataLoader
from cnn_hypertuner import CNNHyperTuner

train_loader = DataLoader(your_train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(your_val_dataset,   batch_size=128)

tuner = CNNHyperTuner(
    algorithm="pso",
    dataset="custom",
    n_iterations=30,
    custom_train_loader=train_loader,
    custom_val_loader=val_loader,
)
result = tuner.run()
```

---

## Algorithm-Specific Tuning

```python
# GA with custom settings
tuner = CNNHyperTuner(
    algorithm="ga",
    dataset="cifar10",
    n_iterations=40,
    algorithm_kwargs={
        "population_size": 30,
        "mutation_rate": 0.2,
        "adaptive_mutation": True,
        "stagnation_restart_after": 15,
    }
)

# PSO with ring topology (more diverse)
tuner = CNNHyperTuner(
    algorithm="pso",
    algorithm_kwargs={
        "n_particles": 30,
        "topology": "ring",
        "w_max": 0.9,
        "w_min": 0.3,
    }
)

# SA with Cauchy perturbation (wider jumps)
tuner = CNNHyperTuner(
    algorithm="sa",
    algorithm_kwargs={
        "T0": 2.0,
        "alpha": 0.95,
        "perturbation": "cauchy",
    }
)
```

---

## Hyperparameter Search Space

| Parameter | Type | Range/Choices |
|---|---|---|
| `learning_rate` | float (log) | [1e-5, 1e-1] |
| `batch_size` | categorical | 16, 32, 64, 128, 256 |
| `num_filters_1/2/3` | int | [16–128], [32–256], [64–512] |
| `kernel_size` | categorical | 3, 5, 7 |
| `dropout_rate` | float | [0.0, 0.7] |
| `dense_units` | int | [64, 1024] |
| `optimizer` | categorical | adam, sgd, rmsprop, adamw |
| `weight_decay` | float (log) | [1e-6, 1e-2] |
| `activation` | categorical | relu, leaky_relu, elu, gelu |
| `num_conv_layers` | int | [2, 5] |

---


## Output

Every run produces:
- `experiments/<name>_results.json` — best config + convergence data
- `experiments/<name>_checkpoint.json` — live best config (updated each eval)
- `experiments/<name>.log` — full training log

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- numpy ≥ 1.21
- matplotlib (optional, for plots)
