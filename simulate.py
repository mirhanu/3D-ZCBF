# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
simulate.py


A script to set up and run CBF-based simulations for 
leader-follower multi-agent systems (MAS).


Author: Mirhan Urkmez
Created: 2025-11-11
"""

import numpy as np
import pickle
from src.leader_follower_mas import LeaderFollowerMAS
from src.utils import *
from src.cbf_control import *
from config.config import *


def prepare_initial_state(dim='2d'):
    if dim == '2d':
        state = np.array(
            [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5]).reshape(-1, 1)
        desired_pos = np.array([[1, 1], [5, 5]])
    else:
        state = np.array([-0.5, -1.0, 1.5, 2.0]).reshape(-1, 1)
        desired_pos = np.atleast_2d(np.array([1, 5])).T
    return state, desired_pos


def setup_mas(config, init_state):
    return LeaderFollowerMAS(
        G=create_communication_graph(config.graph.edges),
        leaders=config.agents.leaders,
        d_max=config.agents.d_max,
        desired_dists=config.graph.desired_dists,
        dt=config.simulation.dtSim,
        state=init_state
    )


def load_jacobian_bounds(is_train, mas, states_array, controls_array, cbfdots_array, save_file):
    if is_train:
        return train_jacobian_bounds(mas, states_array, controls_array, cbfdots_array, save_file=save_file)
    else:
        with open(save_file, 'rb') as f:
            return pickle.load(f)


def control_law_wrapper(state, t, mas, Jubounds, Jlbounds, filtered_states, filtered_controls, filtered_cbfdots, desired_pos, config):
    """CBF control law wrapper."""
    return cbf_control_law(
        state, t, mas, Jubounds, Jlbounds,
        filtered_states, filtered_controls, filtered_cbfdots,
        desired_pos, config.control.k_p, config.control.alpha, config.control.beta_dict
    )


def main():
    # Configuration
    dimension = '1d'  # change to '1d' for 1d simulation results
    is_train = False
    is_save = True
    save_dir = "figures"

    # Available configurations
    configs = {
        '2d': config_2d,
        '1d': config_1d
    }

    # Initial states
    init_states_dict = {
        '2d': [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 1.5],
        '1d': [-0.5, -1.0, 1.5, 2.0]
    }

    # Desired positions
    desired_pos_dict = {
        '2d': [[1, 1], [5, 5]],
        '1d': [[1], [5]]
    }

    # Select config based on dimension
    config = configs[dimension]
    # Prepare initial state and desired positions
    init_state = np.array(init_states_dict[dimension]).reshape(-1, 1)
    desired_pos = np.atleast_2d(desired_pos_dict[dimension])

    # Create MAS
    mas = setup_mas(config, init_state)

    # Generate data
    states_array, controls_array, cbfs_array, cbfdots_array = generate_data(
        mas, config)

    # Filter data satisfying CBF
    filtered_states, filtered_controls, filtered_cbfs, filtered_cbfdots = filter_cbf_satisfied_points(
        states_array, controls_array, cbfs_array, cbfdots_array
    )

    # Load or train Jacobian bounds
    Jubounds, Jlbounds = load_jacobian_bounds(
        is_train, mas, states_array, controls_array, cbfdots_array, config.simulation.jbounds_file
    )

    # Run simulation
    mas.dt = config.simulation.dt
    mas.state = init_state

    def control_fn(s, t): return control_law_wrapper(
        s, t, mas, Jubounds, Jlbounds, filtered_states, filtered_controls, filtered_cbfdots, desired_pos, config
    )

    states, controls = mas.simulate(
        control_law=control_fn, T=config.simulation.T)

    # Plot results
    plot_all_results(states, controls, mas, desired_pos,
                     config, is_save=is_save, save_dir=save_dir)


if __name__ == "__main__":
    main()
