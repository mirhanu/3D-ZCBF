# Distributed Data-Driven Control Barrier Functions for Leader-Follower Multi-Agent Systems

This repository contains code accompanying the paper:

**"Distributed Data-Driven Control Barrier Functions for Leader-Follower Multi-Agent Systems"**  


The code implements **distributed data-driven control barrier functions (3D-ZCBFs)** for connectivity preservation in leader-follower multi-agent systems (MAS) using only local state information.

---

## Abstract

This work introduces distributed data-driven control barrier functions (3D-ZCBFs) for connectivity preservation in unknown control-affine leader–follower MAS. The framework identifies CBF time derivative bounds from collected system data, eliminating the need to model full high-dimensional agent and neighbor dynamics. The learned bounds are then used to ensure controlled invariance of the safe sets. Specific CBFs are designed to preserve connectivity among neighbors, and the conditions are reformulated so that each leader computes its input using only local information. A projection-based controller enforces the 3D-ZCBF constraints while minimally deviating from a nominal input. Simulations demonstrate distributed connectivity preservation using only local information.

---

## 📂 Project Structure
```
3D-ZCBF/
│
├── simulate.py # Main script to run 1D/2D MAS simulations
│
├── src/ # Core code
│ ├── init.py
│ ├── leader_follower_mas.py # MAS dynamics and CBF computation
│ ├── cbf_control_law.py # Data-driven CBF-based control law
│ ├── utils.py # Utility functions: plotting, data generation, nominal control, CBF filtering
│ └── dynamic_system.py # Optional: additional system dynamics or helpers
│
├── config/ # Configuration
│ ├── init.py
│ ├── config.py # Default simulation parameters
│ └── config_struct.py # Dataclasses for configuration structures
│
├── jacobian_bounds/ # Precomputed Jacobian bound files
│ ├── Jbounds1D.pkl
│ └── Jbounds2D.pkl
│
├── figures/ # Simulation results (plots, figures)
│
├── requirements.txt # Python dependencies
└── README.md
```

## Installation

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## 📖 Usage

Run simulations with:

```bash
python simulate.py
```

## Simulation Options

Modify the following variables directly in `simulate.py`:

```python
# Set simulation dimension: '2d' for 2D, '1d' for 1D
dimension = '2d'

# Control whether to train Jacobian bounds or load precomputed ones
is_train = False  # True to train, False to load from jacobian_bounds/

# Control whether to save simulation results or not
is_save = False

# Directory to save simulation results and figures
save_dir = "figures"
```

## 📜 Citing this Work

This repository is for academic and research purposes. Please cite the above paper if used.
