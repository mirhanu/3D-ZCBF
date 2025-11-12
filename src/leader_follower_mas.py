# -*- coding: utf-8 -*-
"""
Leader-Follower Multi-Agent System (MAS) Simulation Module

This module defines the `LeaderFollowerMAS` class, which models a
leader-follower multi-agent system with first-order consensus dynamics.
The system supports:

- Leader and follower agents on an arbitrary communication graph (networkx Graph).
- Desired relative distances between agents.
- Control Barrier Functions (CBFs) for safety constraints.
- Estimation of CBF time derivatives (hdots) and Jacobian bounds.

Dynamics:
------------
Followers:  p_dot_i = -Σ_j∈N(i) (p_i - p_j - d_ij_des)
Leaders:    p_dot_i = -Σ_j∈N(i) (p_i - p_j - d_ij_des) + u_i

"""

import numpy as np
import networkx as nx
import random
from src.dynamic_system import DynamicSystem
from src.utils import estimate_jacobians


class LeaderFollowerMAS(DynamicSystem):
    """
    Leader-Follower Multi-Agent System (first-order consensus dynamics).

    Dynamics:
        Followers:  p_dot_i = -Σ_j∈N(i) (p_i - p_j - d_ij_des)
        Leaders:    p_dot_i = -Σ_j∈N(i) (p_i - p_j - d_ij_des) + u_i

    Args:
        G (nx.Graph): Communication graph (undirected or directed).
        leaders (list[int]): Indices of leader nodes.
        desired_dists (dict): Desired relative distances {(i, j): d_ij_des}.
        dt (float): Integration time step.
        state (np.ndarray): Initial positions of all agents, shape (N,).
    """

    def __init__(self, G, leaders, d_max, desired_dists, dt=0.01, state=None):
        self.G = G
        self.N = len(G.nodes)
        self.leaders = np.array(leaders, dtype=int)
        self.d_max = d_max
        self.followers = np.array(
            [i for i in range(self.N) if i not in self.leaders])
        self.desired_dists = desired_dists  # dictionary {(i, j): d_des_ij}
        self.neighbors_dict = {node: list(
            G.neighbors(node)) for node in G.nodes}
        if state is not None:
            self.n_state = int(state.size/self.N)
            self.n_input = int(state.size/self.N)
        else:
            self.n_state = 1
            self.n_input = 1
        self. initialize_leaders()

        n = self.N*self.n_state  # one state variable per agent (1D position)
        m = len(self.leaders)*self.n_input  # one control input per leader
        super().__init__(n=n, m=m, dt=dt,
                         state=state if state is not None else np.zeros(n))

    def dynamics(self, state, u):
        """
        Compute p_dot for all agents.

        Args:
            state (np.ndarray): Positions of agents, shape (N,).
            u (np.ndarray): Control inputs for leaders, shape (nL,).

        Returns:
            np.ndarray: p_dot (time derivative of positions).
        """
        p_dot = np.zeros(self.n)

        # Compute for all agents
        for i in range(self.N):
            sum_term = 0.0
            slc_i = slice(i * self.n_state,
                          (i + 1) * self.n_state)
            for j in self.G.neighbors(i):
                d_des_ij = self.desired_dists.get(
                    (i, j), -self.desired_dists.get((j, i), 0.0))
                slc_j = slice(j * self.n_state,
                              (j + 1) * self.n_state)
                sum_term += (state[slc_i] - state[slc_j] - d_des_ij)

            # Followers
            if i in self.followers:
                p_dot[slc_i] = -sum_term
            # Leaders
            else:
                leader_idx = np.where(self.leaders == i)[0][0]
                slc_l = slice(leader_idx * self.n_input,
                              (leader_idx + 1) * self.n_input)

                p_dot[slc_i] = -sum_term + u[slc_l]

        return p_dot

    def initialize_leaders(self):
        # Initialize storage for constant leaders if not already done
        if not hasattr(self, "_ff_edge_leaders"):
            self._ff_edge_leaders = {}  # maps (k,j) -> (leader_k, leader_j)
            for (k, j) in self.G.edges:

                if k in self.followers and j in self.followers:
                    neighbors_k = [n for n in self.G.neighbors(
                        k) if n in self.leaders]
                    neighbors_j = [n for n in self.G.neighbors(
                        j) if n in self.leaders]

                    possible_pairs = []

                    for lk in neighbors_k or [None]:
                        for lj in neighbors_j or [None]:
                            # allow lk == lj if j has only one neighbor
                            if lk != lj or len(neighbors_j) == 1:
                                possible_pairs.append((lk, lj))

                    if possible_pairs:
                        leader_k, leader_j = random.choice(possible_pairs)
                    else:
                        # no valid pair, assign None
                        leader_k, leader_j = None, None

                    self._ff_edge_leaders[(k, j)] = (leader_k, leader_j)

    def get_edge_type(self, edge):
        i, j = edge
        if i in self.leaders and j in self.leaders:
            return 'll'
        elif i not in self.leaders and j not in self.leaders:
            return 'ff'
        elif i in self.leaders and j not in self.leaders:
            return 'lf'
        elif i not in self.leaders and j in self.leaders:
            return 'lf'
        else:
            return None

    def cbfdot(self, state, control, edge):
        k, j = edge
        p_dot = self.dynamics(state, control)

        # Slices for agents k and j
        slc_k = slice(k * self.n_state, (k + 1) * self.n_state)
        slc_j = slice(j * self.n_state, (j + 1) * self.n_state)
        state_k, state_j = state[slc_k], state[slc_j]
        dot_k, dot_j = p_dot[slc_k], p_dot[slc_j]

        edge_type = self.get_edge_type(edge)

        if edge_type in ('ll', 'lf'):
            # Simple leader-leader or leader-follower edge
            delta = state_k - state_j
            delta_dot = dot_k - dot_j
            hdots = np.array([-2 * np.dot(delta, delta_dot)])

        elif edge_type == 'ff':
            # Follower-follower edge, depends on leaders
            leader_k, leader_j = self._ff_edge_leaders[edge]

            slc_lk = slice(leader_k * self.n_state,
                           (leader_k + 1) * self.n_state)
            slc_lj = slice(leader_j * self.n_state,
                           (leader_j + 1) * self.n_state)
            state_lk, state_lj = state[slc_lk], state[slc_lj]
            dot_lk, dot_lj = p_dot[slc_lk], p_dot[slc_lj]

            v, w = state_k - state_j, state_lk - state_lj
            v_dot, w_dot = dot_k - dot_j, dot_lk - dot_lj

            vTw = np.dot(v, w)
            w_norm = np.dot(w, w)
            vdot_w = np.dot(v_dot, w)
            v_wdot = np.dot(v, w_dot)
            w_wdot = np.dot(w, w_dot)

            # Parallel and perpendicular components
            h_parallel_dot = -2 * \
                ((vTw * (vdot_w + v_wdot) - (vTw**2) * w_wdot / w_norm) / w_norm)
            h_perp_dot = -2 * np.dot(v, v_dot) - h_parallel_dot

            hdots = np.array([h_parallel_dot, h_perp_dot])

        else:
            hdots = np.array([])  # fallback in case of unknown edge type

        return hdots

    def cbf(self, x_k, x_j, k, j, x_leader_k=None, x_leader_j=None):
        """
        Compute the control barrier function (CBF) for a given edge, automatically determining the type.

        Parameters
        ----------
        x_k : float or np.array
            State of agent k
        x_j : float or np.array
            State of agent j
        k : int
            Index of agent k
        j : int
            Index of agent j
        d_max : float
            Maximum allowed distance
        x_leader_k : float or np.array, optional
            Leader state corresponding to follower k (required for ff)
        x_leader_j : float or np.array, optional
            Leader state corresponding to follower j (required for ff)

        Returns
        -------
        cbf_values : float or tuple
            CBF value(s) depending on edge type
        """
        # Determine roles from class attributes
        role_k = 'l' if k in self.leaders else 'f'
        role_j = 'l' if j in self.leaders else 'f'

        # Decide edge type
        if role_k == 'l' and role_j == 'f':
            edge_type = 'lf'
        elif role_k == 'l' and role_j == 'l':
            edge_type = 'll'
        elif role_k == 'f' and role_j == 'f':
            edge_type = 'ff'
        elif role_k == 'f' and role_j == 'l':
            # Treat as lf but swap positions
            edge_type = 'lf'
            x_k, x_j = x_j, x_k
        else:
            raise ValueError("Invalid roles for agents k and j.")

        # Compute CBF
        if edge_type in ['lf', 'll']:
            x_bar = x_k - x_j
            h = self.d_max**2 - np.dot(x_bar, x_bar)
            return h
        elif edge_type == 'ff':
            if x_leader_k is None or x_leader_j is None:
                raise ValueError(
                    "Leader states required for follower-follower edges.")

            x_bar = x_k - x_j
            x_bar_leaders = x_leader_k - x_leader_j
            eps = 1e-8
            x_bar_hat = x_bar_leaders / (np.linalg.norm(x_bar_leaders)+eps)

            x_bar_parallel = np.dot(x_bar, x_bar_hat) * x_bar_hat
            x_bar_perp = x_bar - x_bar_parallel

            h_parallel = self.d_max**2 / 2 - \
                np.dot(x_bar_parallel, x_bar_parallel)
            h_perp = self.d_max**2 / 2 - np.dot(x_bar_perp, x_bar_perp)

            return h_parallel, h_perp

    def calculate_cbfdots(self, states, controls):
        states = np.atleast_2d(states)  # ensures 2D
        controls = np.atleast_2d(controls)  # ensures 2D
        if states.shape[0] == 1:     # shape (1, 4) or (1, N) → transpose
            states = states.T

        N, steps = states.shape
        cbfdots = {}
        for edge in self.G.edges:
            if self.get_edge_type(edge) == 'ff':
                cbfdots[edge] = np.zeros((steps, 2))
            else:
                cbfdots[edge] = np.zeros((steps, 1))
        for t_step in range(steps):
            x = states[:, t_step]
            u = controls[:, t_step]
            for edge in self.G.edges:
                cbfdots[edge][t_step, :] = self.cbfdot(x, u, edge)

        return cbfdots

    def calculate_cbfs(self, states):
        """
        Compute CBFs for a given state trajectory.

        Parameters
        ----------
        states : np.ndarray
            State trajectory from MAS simulation, shape (steps, N)
            Map agent index -> 'l' or 'f'

        Returns
        -------
        cbfs : dict
            cbfs[(k,j)] = array of shape (steps,2)
            Second column is NaN for edges that are not follower-follower.
        """
        states = np.atleast_2d(states)  # ensures 2D
        if states.shape[0] == 1:     # shape (1, 4) or (1, N) → transpose
            states = states.T

        N, steps = states.shape
        cbfs = {}

        for (k, j) in self.G.edges:
            if k in self.followers and j in self.followers:
                cbfs[(k, j)] = np.zeros((steps, 2))
            else:
                cbfs[(k, j)] = np.zeros((steps, 1))

        for t_step in range(steps):
            x = states[:, t_step]

            for (k, j) in self.G.edges:
                slc_k = slice(k * self.n_state,
                              (k + 1) * self.n_state)
                slc_j = slice(j * self.n_state,
                              (j + 1) * self.n_state)
                x_k = x[slc_k]
                x_j = x[slc_j]

                # Default leaders None
                x_leader_k = None
                x_leader_j = None

                # Only pick leader neighbors if both are followers
                if k in self.followers and j in self.followers:
                    leader_k, leader_j = self._ff_edge_leaders[(k, j)]
                    slc_l_k = slice(leader_k * self.n_state,
                                    (leader_k + 1) * self.n_state)
                    slc_l_j = slice(leader_j * self.n_state,
                                    (leader_j + 1) * self.n_state)
                    if leader_k is not None:
                        x_leader_k = x[slc_l_k]
                    if leader_j is not None:
                        x_leader_j = x[slc_l_j]

                # Compute CBF
                h = self.cbf(x_k, x_j, k, j, x_leader_k, x_leader_j)

                if k in self.followers and j in self.followers:
                    cbfs[(k, j)][t_step, 0] = h[0]
                    cbfs[(k, j)][t_step, 1] = h[1]
                else:
                    cbfs[(k, j)][t_step] = h  # 1D array

        return cbfs

    def estimate_hdots(self, cbfs):
        """
        Estimate time derivatives of CBFs using finite differences.

        Parameters
        ----------
        cbfs : dict
            Dictionary of CBF time series.
            For edge (k,j): cbfs[(k,j)] has shape (steps,) or (steps, 2) for follower-follower edges.

        Returns
        -------
        hdots : dict
            Dictionary of time derivatives of CBFs, same shape as cbfs.
        """
        dt = self.dt
        hdots = {}

        for edge, h_values in cbfs.items():
            # central difference along time (axis=0)
            hdot = (h_values[2:] - h_values[:-2]) / (2 * dt)
            hdots[edge] = hdot

        return hdots

    def estimate_hdots_jacobian_bounds(self, hdots, states, controls, eps_s=0.0, eps_v=0.0):
        """
        Estimate upper and lower Jacobian bounds for CBF derivatives,
        considering both states and control inputs.

        Parameters
        ----------
        hdots : dict
            Dictionary of CBF derivative time series, shape (steps,) or (steps,2)
        states : np.ndarray
            State trajectory, shape (steps, N_agents*n_state)
        controls : np.ndarray
            Control trajectory, shape (steps-1, M)
        eps_s : float or array
            Bound on state/control measurement noise
        eps_v : float or array
            Bound on hdots measurement noise

        Returns
        -------
        Ju_dict : dict
            Upper Jacobian bounds for each edge
        Jl_dict : dict
            Lower Jacobian bounds for each edge
        """
        Ju_dict = {}
        Jl_dict = {}

        for edge, hdot_values in hdots.items():
            k, j = edge

            # Determine edge type
            if k in self.leaders and j in self.leaders:
                edge_type = 'll'
            elif k in self.leaders or j in self.leaders:
                edge_type = 'lf'
            else:
                edge_type = 'ff'

            # Build slices for states
            state_slices = []
            nodes_to_include = [
                k, j] + list(self.G.neighbors(k)) + list(self.G.neighbors(j))
            if edge_type == 'ff':
                leader_k, leader_j = self._ff_edge_leaders[(k, j)]
                nodes_to_include += [leader_k, leader_j] + list(
                    self.G.neighbors(leader_k)) + list(self.G.neighbors(leader_j))
            elif edge_type == 'lf':
                leader_node = k if k in self.leaders else j
                nodes_to_include += [leader_node]
            # Convert nodes to slices
            nodes_to_include = set(nodes_to_include)
            for node in nodes_to_include:
                slc = slice(node*self.n_state, (node+1)*self.n_state)
                state_slices.append(slc)

            # Build slices for controls
            control_slices = []
            if edge_type == 'll':
                control_slices = [slice(idx*self.n_input, (idx+1)*self.n_input)
                                  for idx in [np.where(self.leaders == k)[0][0],
                                              np.where(self.leaders == j)[0][0]]]
            elif edge_type == 'lf':
                leader_node = k if k in self.leaders else j
                control_slices = [slice(np.where(self.leaders == leader_node)[0][0]*self.n_input,
                                        (np.where(self.leaders == leader_node)[0][0]+1)*self.n_input)]
            elif edge_type == 'ff':
                leader_k_idx = np.where(self.leaders == leader_k)[0][0]
                leader_j_idx = np.where(self.leaders == leader_j)[0][0]
                control_slices = [slice(leader_k_idx*self.n_input, (leader_k_idx+1)*self.n_input),
                                  slice(leader_j_idx*self.n_input, (leader_j_idx+1)*self.n_input)]

            # Stack relevant state and control data
            s_input = np.hstack([states[:, slc] for slc in state_slices] +
                                [controls[:, slc] for slc in control_slices])

            # Use next-step hdots as outputs
            y_data = hdot_values
            if hdot_values.ndim == 1:
                y_data = y_data.reshape(-1, 1)

            # Estimate Jacobian bounds
            Ju, Jl = estimate_jacobians(
                s_input, y_data, eps_s=eps_s, eps_v=eps_v)

            # Map Jacobians back to full state+control dimensions
            total_vars = self.n + self.m
            num_components = hdot_values.shape[1] if hdot_values.ndim > 1 else 1
            full_Ju = np.zeros((num_components, total_vars))
            full_Jl = np.zeros((num_components, total_vars))

            # Collect dependent indices
            dependent_indices = []
            for slc in state_slices:
                dependent_indices.extend(range(slc.start, slc.stop))
            for slc in control_slices:
                dependent_indices.extend(
                    range(self.n + slc.start, self.n + slc.stop))

            full_Ju[:, dependent_indices] = Ju
            full_Jl[:, dependent_indices] = Jl

            Ju_dict[edge] = full_Ju
            Jl_dict[edge] = full_Jl

        return Ju_dict, Jl_dict
