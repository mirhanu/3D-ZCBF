# -*- coding: utf-8 -*-
"""
cbf_control_law.py


Implementation of the data driven Control Barrier Function (CBF)
based control law for a leader–follower multi-agent system (MAS).


Created on: 2025-11-11
Author: Mirhan Urkmez


This file contains:
- `cbf_control_law(...)` : convex-program-based controller that solves for
agent inputs while respecting robust CBF constraints using dataset-derived
Jacobian bounds.
- `select_data_index_for_edge(...)`: helper that selects the dataset index
(i*) for a given edge.

Notes
-----
- The code expects the `mas` object to provide certain fields and methods
(see docstring of `cbf_clean_control_law` for details).
- This module does not change algorithmic behaviour, only improves
readability, documentation and defensive checks for publishing.


"""

from typing import Dict, Iterable, List, Optional, Tuple
import cvxpy as cp
import numpy as np
from src.utils import nominal_controller


def cbf_control_law(state, t, mas, Ju_dict, Jl_dict, states_array, controls_array, cbfdots_array, desired_pos, k_p=0.5, alpha=100.0, beta_dict=None, slack_weight=1e9):
    """
    Compute control inputs by solving a convex CBF optimization problem.


    Parameters
    ----------
    state : np.ndarray
    Current full system state vector (stacked agent states).
    t : float
    Current time (not used directly by this routine, included for API
    compatibility).
    mas : LeaderFollowerMAS
    Multi-agent system object with (at least) the following attributes /
    methods used here:
    - G.edges: iterable of edges (tuples)
    - leaders: iterable/array of leader agent indices
    - n_state, n_input, n, m: per-agent / global dimensional information
    - neighbors_dict: dict mapping agent -> list of neighbor agent indices
    - get_edge_type(edge): returns string in {'ll','lf','ff'}
    - calculate_cbfs(state): returns mapping edge -> (h_vec, ...) where h
    is used to compute the exponential term
    Ju_dict, Jl_dict : dict
    Mapping edge -> upper and lower Jacobian bounds (numpy arrays).
    states_array : np.ndarray
    Dataset states of shape (n_total_state, N_data).
    controls_array : np.ndarray
    Dataset inputs of shape (n_total_input, N_data).
    cbfdots_array : dict
    Mapping edge -> array of CBF derivatives on the dataset with shape
    (N_data, n_cbfs_for_edge).
    desired_pos : np.ndarray
    Desired positions used by the nominal controller.
    k_p, alpha, beta_dict, slack_weight : parameters
    Tuning parameters used in cost and constraints (see code comments).


    Returns
    -------
    u.value : np.ndarray
    Optimized control input vector (length = mas.m). If the solver fails
    this function will raise an exception from the solver or return None.


    """

    # Nominal controller and CVXPY setup
    u_nom = nominal_controller(state, mas, desired_pos, k_p)
    n_u = mas.m
    n_x = mas.n

    u = cp.Variable(n_u)

    # list to collect all slack variables
    slack_vars = []

    # quadratic tracking cost to nominal control
    cost = cp.sum_squares(u-u_nom.T)
    constraints = []

    # ------------------------------------------------------------------
    # Pre-select i* indices for dataset points for each edge
    # ------------------------------------------------------------------
    i_star_edges = {}
    for edge in mas.G.edges:
        edge_type = mas.get_edge_type(edge)
        Ju = Ju_dict[edge]
        Jl = Jl_dict[edge]
        i_star_edges[edge] = select_data_index_for_edge(
            edge, state, states_array, cbfdots_array, Ju, Jl, mas
        )

    # ------------------------------------------------------------------
    # Build constraints for every CBF corresponding to every edge
    # ------------------------------------------------------------------
    for edge in mas.G.edges:
        edge_type = mas.get_edge_type(edge)
        i_star_list = i_star_edges[edge]
        n_cbfs = len(i_star_list)
        k, j = edge
        Ju = Ju_dict[edge]
        Jl = Jl_dict[edge]

        # determine leaders for this edge
        if edge_type == 'll':
            leaders = list(edge)
        elif edge_type == 'lf':
            leaders = [k if k in mas.leaders else j]  # single leader
        elif edge_type == 'ff':
            leaders = mas._ff_edge_leaders[edge]  # tuple of two leaders

        # Followers only relevant for FF edges
        followers = list(edge) if edge_type == 'ff' else []

        # Compute shared neighbors among leaders
        leader_neighbors = [mas.neighbors_dict[l] for l in leaders]
        if len(leaders) > 1:
            shared_neighbors = set(leader_neighbors[0])
            for ln in leader_neighbors[1:]:
                shared_neighbors &= set(ln)
        else:
            shared_neighbors = set()  # single leader -> no shared neighbors

        # Loop over CBFs for this edge
        for cbf_idx, i_star in enumerate(i_star_list):
            hx = mas.calculate_cbfs(state)[edge][0, cbf_idx]
            i_star = i_star_list[cbf_idx]
            delta_x = state - states_array[:, i_star]
            delta_x_plus = np.maximum(delta_x, 0)
            delta_x_minus = np.maximum(-delta_x, 0)
            hdot_i = cbfdots_array[edge][i_star, cbf_idx]

            # Loop over leaders
            for l in leaders:
                # --- input slice & delta variables ---
                idx_l = np.where(mas.leaders == l)[0][0]
                slc_input = slice(idx_l * mas.n_input, (idx_l+1) * mas.n_input)
                delta_u_plus = cp.Variable(mas.n_input, nonneg=True)
                delta_u_minus = cp.Variable(mas.n_input, nonneg=True)
                constraints += [u[slc_input] - controls_array[slc_input,
                                                              i_star] == delta_u_plus - delta_u_minus]

                # --- state slice for own agent ---
                slc_state = slice(l*mas.n_state, (l+1)*mas.n_state)
                if beta_dict is not None and (edge_type in ['ll', 'ff']):
                    # use the beta specified for this edge and leader
                    beta_l = beta_dict.get((edge, l), 1.0 / len(leaders))
                else:
                    # single leader (lf) or no dict provided
                    beta_l = 1.0

                # --- compute LHS ---
                lhs = beta_l * hdot_i + \
                    Jl[cbf_idx, n_x + np.arange(slc_input.start, slc_input.stop)] @ delta_u_plus - \
                    Ju[cbf_idx, n_x + np.arange(slc_input.start, slc_input.stop)] @ delta_u_minus + \
                    Jl[cbf_idx, slc_state] @ delta_x_plus[slc_state] - \
                    Ju[cbf_idx, slc_state] @ delta_x_minus[slc_state]

                # --- add neighbors (exclude shared for beta) ---
                neighbors = mas.neighbors_dict[l]
                for n in neighbors:
                    if n not in shared_neighbors and n not in leaders and n not in followers:
                        slc_n = slice(n*mas.n_state, (n+1)*mas.n_state)
                        lhs += Jl[cbf_idx, slc_n] @ delta_x_plus[slc_n] - \
                            Ju[cbf_idx, slc_n] @ delta_x_minus[slc_n]

                # --- add shared neighbors with beta ---
                for n in shared_neighbors:
                    if n not in leaders and n not in followers:
                        slc_n = slice(n*mas.n_state, (n+1)*mas.n_state)
                        lhs += beta_l*(Jl[cbf_idx, slc_n] @ delta_x_plus[slc_n] -
                                       Ju[cbf_idx, slc_n] @ delta_x_minus[slc_n])

                # ---- FF edges: include only this leader's own follower ----
                if edge_type == 'ff':
                    l1, l2 = mas._ff_edge_leaders[edge]
                    if l == l1:
                        f = edge[0]
                        slc_f = slice(f * mas.n_state, (f + 1) * mas.n_state)
                        lhs += Jl[cbf_idx, slc_f] @ delta_x_plus[slc_f] - \
                            Ju[cbf_idx, slc_f] @ delta_x_minus[slc_f]
                    elif l == l2:
                        f = edge[1]
                        slc_f = slice(f * mas.n_state, (f + 1) * mas.n_state)
                        lhs += Jl[cbf_idx, slc_f] @ delta_x_plus[slc_f] - \
                            Ju[cbf_idx, slc_f] @ delta_x_minus[slc_f]

                # --- slack variable ---
                s = cp.Variable(nonneg=True)
                slack_vars.append(s)
                constraints += [lhs + s >= -beta_l * alpha * np.exp(hx)]
    if slack_vars:
        cost += slack_weight * cp.sum_squares(cp.hstack(slack_vars))

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=True)

    return u.value


def select_data_index_for_edge(edge, state, states_array, cbfdots_array, Ju, Jl, mas, eps=1e-4):
    """
    Select dataset index i* (per CBF) for the specified edge.

    The function computes a score for every dataset sample and selects the
    index with maximal score for each CBF of the edge. When no valid sample
    exists it raises a ValueError.
    """

    N = states_array.shape[1]
    n_cbfs = cbfdots_array[edge].shape[1]  # number of cbfs for this edge
    selected_indices = []
    n_state = state.shape[0]

    edge_type = mas.get_edge_type(edge)

    for cbf_idx in range(n_cbfs):
        scores = np.full(N, -np.inf)
        for k in range(N):
            # current dataset state
            delta_x = state-states_array[:, k]
            delta_x_plus = np.maximum(delta_x, 0)
            delta_x_minus = np.maximum(-delta_x, 0)

            # For ff edges we must ensure the dataset point interpolated states
            # are within state set X.
            valid = True
            if edge_type == 'ff':
                leader_k, leader_j = mas._ff_edge_leaders.get(edge)
                slc_lk = slice(leader_k * mas.n_state,
                               (leader_k + 1) * mas.n_state)
                slc_lj = slice(leader_j * mas.n_state,
                               (leader_j + 1) * mas.n_state)

                x_leader_k = state[slc_lk]
                x_leader_j = state[slc_lj]
                x_data_k = states_array[slc_lk, k]
                x_data_j = states_array[slc_lj, k]

                v0 = x_leader_k - x_leader_j      # current state difference
                v1 = x_data_k - x_data_j          # dataset difference
                # critical point along the segment
                diff = v1 - v0
                norm_diff_sq = np.dot(diff, diff)
                if np.linalg.norm(v1) < eps:
                    valid = False
                elif norm_diff_sq > 0:
                    lambda_c = -np.dot(v0, diff) / norm_diff_sq
                    if 0 <= lambda_c <= 1:
                        v_c = v0 + lambda_c * diff
                        if np.linalg.norm(v_c) < eps:
                            valid = False

            if valid:
                hdot_val = cbfdots_array[edge][k, cbf_idx]
                scores[k] = hdot_val + \
                    np.sum(Jl[cbf_idx, :n_state] @ delta_x_plus) - \
                    np.sum(Ju[cbf_idx, :n_state] @ delta_x_minus)

        # select dataset point with max score
        if np.all(scores == -np.inf):
            raise ValueError(
                f"No valid dataset point found for edge {edge}, CBF {cbf_idx}")

        i_star = int(np.argmax(scores))
        selected_indices.append(i_star)

    return selected_indices
