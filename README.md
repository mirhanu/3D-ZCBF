# Distributed Data-Driven Control Barrier Functions (3D-ZCBFs)

The code implements **distributed data-driven control barrier functions (3D-ZCBFs)** for connectivity preservation in leader-follower multi-agent systems (MAS) using only local state information.


The code implements **distributed data-driven control barrier functions (3D-ZCBFs)** for connectivity preservation in leader-follower multi-agent systems (MAS) using only local state information.

---

## Methodology Overview

The 3D-ZCBF framework ensures controlled invariance of safe sets by identifying CBF time derivative bounds directly from collected system data. This approach eliminates the requirement for full modeling of high-dimensional agent and neighbor dynamics. 

Key features of this implementation include:
* **Local Information Only:** Leaders compute control inputs using only local state data.
* **Connectivity Preservation:** Specific CBFs are designed to maintain links between neighbors in the MAS.
* **Projection-Based Control:** A controller enforces 3D-ZCBF constraints while ensuring minimal deviation from a nominal control input.
* **Data-Driven Bounds:** Learned Jacobian bounds are used to guarantee safety without explicit system identification.

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

## 📜 Related Work

If you use this code or the associated methods, please cite the following:

* "A Distributed Framework for Data-Driven Safe Coordination in Leader–Follower Networks" (Under Review, 2026).
*  "Distributed Data-Driven Control Barrier Functions for Leader-Follower Multi-Agent Systems" (Under Review, 2025).