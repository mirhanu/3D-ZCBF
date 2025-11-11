# Distributed Data-Driven Control Barrier Functions for Leader-Follower Multi-Agent Systems

This repository contains code accompanying the paper:

**"Distributed Data-Driven Control Barrier Functions for Leader-Follower Multi-Agent Systems"**  


The code implements **distributed data-driven control barrier functions (3D-ZCBFs)** for connectivity preservation in leader-follower multi-agent systems (MAS) using only local state information.

---

## Abstract

This work introduces distributed data-driven control barrier functions (3D-ZCBFs) for connectivity preservation in unknown control-affine leader–follower MAS. The framework identifies CBF time derivative bounds from collected system data, eliminating the need to model full high-dimensional agent and neighbor dynamics. The learned bounds are then used to ensure controlled invariance of the safe sets. Specific CBFs are designed to preserve connectivity among neighbors, and the conditions are reformulated so that each leader computes its input using only local information. A projection-based controller enforces the 3D-ZCBF constraints while minimally deviating from a nominal input. Simulations demonstrate distributed connectivity preservation using only local information.

---

## Repository Structure

- `simulate.py`  
  Main simulation script. Generates 1D and 2D MAS simulations in the paper.

- `cbf_control_law.py`  
  Implementation of the data-driven CBF-based control law for leader–follower MAS.

- `utils.py`  
  Utility functions for plotting, data generation, nominal control, Jacobian estimation, and CBF filtering.

- `config.py`  
  Configuration file specifying simulation parameters, agent parameters, communication graphs, and control parameters.

- `config_struct.py`  
  Dataclasses defining the configuration structures for simulation, agents, graph, and control.

- `leader_follower_mas.py`  
  Class implementing the MAS dynamics and CBF computations.

---


## Usage

Run simulations with:

```bash
python simulate.py
```

- Modify `dimension = '2d'` or `'1d'` in `simulate.py` to switch between 2D and 1D simulations.
- `is_train = True` to train Jacobian bounds or `False` to load precomputed bounds.
- Simulation results (plots) will be saved to the folder specified by `save_dir`.

---

## License

This repository is for academic and research purposes. Please cite the above paper if used.
