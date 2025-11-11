# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Mirhan Urkmez
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

# ----------------------- Simulation Parameters ----------------------- #


@dataclass
class SimulationConfig:
    dt: float
    dtSim: float
    TSim: float
    T: float
    num_states: int
    jbounds_file: str

# ----------------------- Agent Parameters ---------------------------- #


@dataclass
class AgentConfig:
    leaders: List[int]
    d_max: float
    # x_ranges[i] can be list of tuples (multi-dim), u_ranges same
    x_ranges: dict[int, List[Tuple[float, float]]]
    u_ranges: dict[int, List[Tuple[float, float]]]


# ----------------------- Graph Parameters ---------------------------- #
@dataclass
class GraphConfig:
    edges: List[Tuple[int, int]]
    # desired_dists can be scalar (1D) or array (2D)
    desired_dists: dict[Tuple[int, int], Union[float, np.ndarray]]


# ----------------------- Control Parameters -------------------------- #
@dataclass
class ControlConfig:
    k_p: float
    alpha: float
    beta_dict: dict[Tuple[Tuple[int, int], int], float]


# ----------------------- Full Unified Config ------------------------- #
@dataclass
class Config:
    simulation: SimulationConfig
    agents: AgentConfig
    graph: GraphConfig
    control: ControlConfig
