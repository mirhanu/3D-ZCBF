# -*- coding: utf-8 -*-
"""

config.py

Configuration instances for multi-agent system (MAS) simulations used in the paper.


@author: % Mirhan Urkmez
Created: 2025-11-11

"""

import numpy as np
from config.config_struct import *

# ----------------- 2D configuration example ----------------- #
config_2d = Config(
    simulation=SimulationConfig(
        dt=0.01, dtSim=0.001, TSim=0.1, T=5.0, num_states=50,
        jbounds_file='jacobian_bounds//Jbounds2D.pkl'),
    agents=AgentConfig(
        leaders=[0, 3],
        d_max=3.0,
        x_ranges={i: [(-5, 5), (-5, 5)] for i in range(4)},
        u_ranges={0: [(-5, 5), (-5, 5)], 1: [(-5, 5), (-5, 5)]}
    ),
    graph=GraphConfig(
        edges=[(0, 1), (1, 2), (2, 3)],
        desired_dists={
            (0, 1): np.array([1.0, 2.0]),
            (1, 2): np.array([2.0, 1.0]),
            (2, 3): np.array([1.0, 1.0])
        }
    ),
    control=ControlConfig(
        k_p=10.0,
        alpha=100,
        beta_dict={((1, 2), 0): 0.5, ((1, 2), 3): 0.5}
    )
)

# ----------------- 1D configuration example ----------------- #
config_1d = Config(
    simulation=SimulationConfig(
        dt=0.01, dtSim=0.01, TSim=1.0, T=1.0, num_states=50,
        jbounds_file='jacobian_bounds//Jbounds1D.pkl'),
    agents=AgentConfig(
        leaders=[0, 3],
        d_max=3.0,
        x_ranges={i: [(-5, 5)] for i in range(4)},
        u_ranges={0: [(-5, 5)], 1: [(-5, 5)]}
    ),
    graph=GraphConfig(
        edges=[(0, 1), (0, 2), (2, 3), (0, 3)],
        desired_dists={(0, 1): 1, (0, 2): 1, (2, 3): 1, (0, 3): 1}
    ),
    control=ControlConfig(
        k_p=10.0,
        alpha=10.0,
        beta_dict={((0, 3), 0): 0.5, ((0, 3), 3): 0.5}
    )
)
