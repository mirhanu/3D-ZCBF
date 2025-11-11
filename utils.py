# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Mirhan Urkmez
"""

from matplotlib.lines import Line2D
import numpy as np
import cvxpy as cp
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from sympy import symbols, Matrix, diff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import to_rgb
from matplotlib.collections import LineCollection
import colorsys
from config_struct import *


def create_communication_graph(edges):
    """Create a NetworkX graph from a list of edges."""
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def simulate_mas(mas, init_states_list, control_law, T):
    """Simulate MAS for multiple initial states and compute CBFs."""
    # Initialize empty lists
    all_states_list = []
    all_controls_list = []
    all_cbfs_dict = {edge: [] for edge in mas.G.edges}
    all_cbfdots_dict = {edge: [] for edge in mas.G.edges}

    for init_state in init_states_list:
        mas.state = init_state.copy()

        states, controls = mas.simulate(control_law, T=T)
        cbfs = mas.calculate_cbfs(states)
        cbfdots = mas.estimate_hdots(cbfs)

        # Append to lists
        all_states_list.append(states[:, 1:-1])
        all_controls_list.append(controls[:, 1:])

        for edge in mas.G.edges:
            cbf_edge = cbfs[edge][1:-
                                  1]
            all_cbfs_dict[edge].append(cbf_edge)

            cbfdot_edge = cbfdots[edge]
            all_cbfdots_dict[edge].append(cbfdot_edge)

    # Convert lists to numpy arrays after loop
    all_states_array = np.hstack(all_states_list)
    all_controls_array = np.hstack(all_controls_list)
    all_cbfs_array = {edge: np.vstack(
        all_cbfs_dict[edge]) for edge in mas.G.edges}
    all_cbfdots_array = {edge: np.vstack(
        all_cbfdots_dict[edge]) for edge in mas.G.edges}
    return all_states_array, all_controls_array, all_cbfs_array, all_cbfdots_array


def plot_cbfs(cbfs, edges, dt, save_path=None):
    """Plot CBF evolution over time."""
    edges = list(edges)
    t_array = np.arange(cbfs[edges[0]].shape[0]) * dt
    plt.figure(figsize=(8, 5))
    y_min = np.inf
    y_max = -np.inf

    for edge in edges:
        n_cbf = cbfs[edge].shape[1]
        for i in range(n_cbf):
            h = cbfs[edge][:, i]
            plt.plot(
                t_array,
                cbfs[edge],
                label=rf"$h^{{{edge[0]}{edge[1]}}}_{{{i+1}}}$",
                linewidth=3
            )
            y_min = min(y_min, h.min())
            y_max = max(y_max, h.max())

    y_margin = 0.05 * (y_max - y_min)  # 5% margin
    plt.ylim(bottom=min(0, y_min - y_margin), top=y_max + y_margin)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel("CBF value", fontsize=16)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def lighten_color_rgb(rgb, factor):
    """Lighten color by factor ∈ [0,1]; 0 -> original, 1 -> white, preserving hue via HLS."""
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # increase lightness toward 1
    l_new = l + (1 - l) * factor
    return colorsys.hls_to_rgb(h, l_new, s)


def plot_agent_trajectories(states, dt, n_state=2, save_path=None):
    """
    Plot agent trajectories for possibly multi-dimensional states.

    Parameters
    ----------
    states : np.ndarray
        Array of shape (n_agents * n_state, N).
    dt : float
        Sampling time.
    n_state : int
        Number of state dimensions per agent.
    """
    n_agents = states.shape[0] // n_state
    N = states.shape[1]
    t_array = np.arange(N) * dt

    if n_state == 1:
        plt.figure(figsize=(8, 5))

        for i in range(n_agents):
            plt.plot(
                t_array,
                states[i, :],
                label=rf"$\mathbf{{x}}_{i}$",
                linewidth=3
            )

        plt.xlabel("Time [s]", fontsize=16)
        plt.ylabel("Position", fontsize=16)
        # plt.title("Agent Trajectories", fontsize=16, fontweight='bold')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=14, loc='best')

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
    elif n_state == 2:
        # 2D states: plot trajectories in xy plane
        fig, ax = plt.subplots(figsize=(6, 6))

        base_colors = plt.cm.tab10.colors  # distinct hues

        legend_handles = []
        legend_labels = []

        for i in range(n_agents):
            x = states[i * n_state, :]
            y = states[i * n_state + 1, :]

            # Build segments for LineCollection
            points = np.vstack([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            base = np.array(to_rgb(base_colors[i % len(base_colors)]))

            # Create gradient colors preserving hue (lightness factor from 0->0.9)
            colors = [lighten_color_rgb(base, k / (N - 1) * 0.9)
                      for k in range(N - 1)]

            lc = LineCollection(segments, colors=colors,
                                linewidths=3, zorder=1)
            ax.add_collection(lc)

            # Start / end markers on top
            ax.plot(x[0], y[0], 'o', color=base, markersize=6, zorder=3)
            ax.plot(x[-1], y[-1], 's', color=base, markersize=7, zorder=3)

            # Create a proxy line for legend (single handle per agent)
            proxy = Line2D([0], [0], color=base, lw=3)
            legend_handles.append(proxy)
            legend_labels.append(rf"$\mathbf{{x}}_{{{i}}}$")

        # Make sure axis bounds include all the collections
        ax.autoscale_view()
        # ax.set_aspect('equal', 'box')
        ax.set_aspect(0.7)  # makes x twice as long as y

        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Place legend: 'best' may overlap; consider bbox_to_anchor or loc='upper right'
        ax.legend(legend_handles, legend_labels,
                  fontsize=12, loc='best', framealpha=0.9)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    elif n_state == 3:
        # 3D states: plot trajectories in xyz space
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(n_agents):
            x = states[i * n_state, :]
            y = states[i * n_state + 1, :]
            z = states[i * n_state + 2, :]
            ax.plot(x, y, z, label=rf"$\mathbf{{x}}_{i}$",
                    linewidth=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.title("Agent trajectories (3D)")
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=14, loc='best')

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    else:
        # For higher-dimensional states, plot each dimension vs time
        for i in range(n_agents):
            plt.figure()
            for j in range(n_state):
                plt.plot(
                    t_array, states[i * n_state + j, :],
                    label=rf"$\mathbf{{x}}_{i}$",
                    linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel("State value")
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend(fontsize=14, loc='best')

            if save_path is not None:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()


def plot_nominal_vs_cbf_controller(controls, states, mas, dt, leaders, desired_pos, k_p=0.5,  save_path=None):
    """Plot nominal control vs CBF control of agents."""
    n_agents = len(leaders)
    N = controls.shape[1]
    u_nom = nominal_controller(states, mas, desired_pos, k_p)
    t_array = np.arange(N) * dt
    plt.figure()
    base_colors = plt.cm.tab10.colors
    for i in range(n_agents):
        slc_k = slice(i * mas.n_input,
                      (i + 1) * mas.n_input)
        for dim in range(mas.n_input):
            color = base_colors[(i*n_agents+dim) % len(base_colors)]
            plt.plot(t_array, controls[slc_k, :][dim, :],
                     label=rf"$\mathbf{{u}}_{{{leaders[i]}}}^{{({dim+1}),*}}$", color=color, linewidth=2)
            plt.plot(t_array, u_nom[slc_k, 0:N][dim, :], linestyle='--',
                     label=rf"$\mathbf{{u}}_{{{leaders[i]}}}^{{({dim+1}),nom}}$", color=color, linewidth=2)
    plt.xlabel("Time [s]", fontsize=16)
    plt.ylabel(r"$\mathbf{u}_i$", fontsize=16, fontweight='bold')

    # Grid and legend
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=14, loc='best')

    # Save figure if path is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_all_results(states, controls, mas, desired_pos, config, is_save=False, save_dir="figures"):
    """Plot all simulation results."""
    h = mas.calculate_cbfs(states)

    edge_str = "-".join([f"{i}{j}" for (i, j) in config.graph.edges])
    dist_str = "-".join([str(config.graph.desired_dists[e])
                        for e in config.graph.edges])

    # Generate filenames
    filename_cbf = f"{save_dir}/cbf_edges_{edge_str}_d_{dist_str}.jpeg"
    filename_state = f"{save_dir}/state_edges_{edge_str}_d_{dist_str}.jpeg"
    filename_control = f"{save_dir}/control_edges_{edge_str}_d_{dist_str}.jpeg"

    # Plot
    plot_cbfs(h, mas.G.edges, mas.dt, save_path=filename_cbf)
    plot_agent_trajectories(states, mas.dt, mas.n_state,
                            save_path=filename_state)
    plot_nominal_vs_cbf_controller(controls, states, mas, mas.dt, mas.leaders,
                                   desired_pos, config.control.k_p, save_path=filename_control)


def estimate_jacobians(s_data, y_data, eps_s=0.0, eps_v=0.0):
    """
    Estimate upper and lower Jacobian bounds J_u and J_l
    given state/input data s_data and next output y_data.

    Args:
        s_data: (N, n_s) array of input/state vectors
        y_data: (N, n_y) array of next output/state vectors
        eps_s: scalar or array of length n_s, bound on s measurement noise
        eps_v: scalar or array of length n_y, bound on y measurement noise

    Returns:
        J_u: (n_y, n_s) upper Jacobian bounds
        J_l: (n_y, n_s) lower Jacobian bounds
    """
    N, n_s = s_data.shape
    _, n_y = y_data.shape

    # Prepare pairs (j, l)
    pairs = list(combinations(range(N), 2))

    n_pairs = len(pairs)

    delta_s_plus = np.zeros((n_pairs, n_s))
    delta_s_minus = np.zeros((n_pairs, n_s))
    delta_y = np.zeros((n_pairs, n_y))

    for idx, (j, l) in enumerate(pairs):
        ds = s_data[j] - s_data[l]
        delta_s_plus[idx] = np.maximum(ds, 0)
        delta_s_minus[idx] = -np.minimum(ds, 0)
        delta_y[idx] = y_data[j] - y_data[l]

    # Decision variables: Ju and Jl (n_y x n_s)
    Ju = cp.Variable((n_y, n_s))
    Jl = cp.Variable((n_y, n_s))

    constraints = []

    for k in range(n_pairs):
        j, l = pairs[k]
        ds = s_data[j] - s_data[l]
        if np.linalg.norm(ds) > 0.1:
            constraints.append(
                delta_y[k] <= Ju @ delta_s_plus[k] - Jl @ delta_s_minus[k] + 2*eps_v)
            constraints.append(
                delta_y[k] >= Jl @ delta_s_plus[k] - Ju @ delta_s_minus[k] - 2*eps_v)
    constraints.append(Ju >= Jl)

    # Objective: sum of widths across all outputs and inputs
    objective = cp.Minimize(
        cp.sum(cp.matmul(Ju - Jl, (delta_s_plus + delta_s_minus).T))
    )

    # Solve LP
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=True)

    return Ju.value, Jl.value


def filter_cbf_satisfied_points(states_array, controls_array, cbfs_array, cbfdots_array, alpha=10.0):
    """
    Keep only points where CBF condition (cbfdot + alpha * cbf >= 0) 
    is satisfied for all edges.

    Parameters
    ----------
    states_array : dict
        Dictionary of states per edge (i,j) -> np.ndarray (N, n_x)
    controls_array : dict
        Dictionary of controls per edge (i,j) -> np.ndarray (N, n_u)
    cbfs_array : dict
        Dictionary of CBF values per edge (i,j) -> np.ndarray (N,)
    cbfdots_array : dict
        Dictionary of CBF derivatives per edge (i,j) -> np.ndarray (N,)
    alpha : float
        Class-K function gain used in condition.

    Returns
    -------
    filtered_states : np.ndarray
        States for points where all CBFs satisfy the condition.
    filtered_controls : np.ndarray
        Controls for points where all CBFs satisfy the condition.
    filtered_cbfs : np.ndarray
        CBF values for those points.
    filtered_cbfdots : np.ndarray
        CBF derivatives for those points.
    """
    # assume all edges have same number of samples
    keys = list(cbfs_array.keys())
    N = len(cbfs_array[keys[0]])

    mask_all = np.ones(N, dtype=bool)

    for k in cbfs_array.keys():
        cbf = cbfs_array[k]
        cbfdot = cbfdots_array[k]
        mask_edge = (cbfdot + alpha * cbf >= 0)
        # If mask_edge is 2D, reduce across columns
        if mask_edge.ndim > 1:
            mask_edge = np.all(mask_edge, axis=1)

        mask_all &= mask_edge

    # Filter states/controls along columns
    filtered_states = states_array[:, mask_all]
    filtered_controls = controls_array[:, mask_all]

    # stack cbf and cbfdot arrays for all edges, for those points
    filtered_cbfs = {k: v[mask_all] for k, v in cbfs_array.items()}
    filtered_cbfdots = {k: v[mask_all] for k, v in cbfdots_array.items()}

    return filtered_states, filtered_controls, filtered_cbfs, filtered_cbfdots


def select_points_kmeans(states, controls, hdot_dict, n_points=50, random_state=0, normalize=True):
    """
    Select diverse samples from state and control histories using KMeans clustering
    in the joint (state, control) space.

    Parameters
    ----------
    states : np.ndarray, shape (N, n_states)
        State history
    controls : np.ndarray, shape (N, n_controls)
        Control history
    hdot_dict : dict
        Dictionary of hdot histories for each edge, e.g. {(i,j): np.ndarray of shape (N,)}
    n_points : int
        Number of diverse samples to select
    random_state : int
        Random seed for reproducibility
    normalize : bool
        If True, normalize joint features before clustering

    Returns
    -------
    selected_states : np.ndarray
    selected_controls : np.ndarray
    selected_hdot_dict : dict
        Dictionary with same keys as hdot_dict, but containing only selected samples
    """
    # Combine states and controls
    X = np.hstack((states, controls))

    # Normalize to balance scales if needed
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # Run KMeans
    kmeans = KMeans(n_clusters=n_points, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)

    # Find representative samples for each cluster
    centers = kmeans.cluster_centers_
    selected_idx = [np.argmin(np.linalg.norm(
        X_scaled - c, axis=1)) for c in centers]
    selected_idx = np.unique(selected_idx)

    # Select corresponding samples
    selected_states = states[selected_idx]
    selected_controls = controls[selected_idx]
    selected_hdot_dict = {edge: hdot[selected_idx]
                          for edge, hdot in hdot_dict.items()}

    return selected_states, selected_controls, selected_hdot_dict


def nominal_controller(states, mas, desired_pos, k_p=0.5):
    """
    Compute nominal control inputs for leader agents in a leader-follower MAS.

    Parameters
    ----------
    states : np.ndarray
        Current agent positions, shape (n_agents,) or (n_agents, dim)
    mas : object
        Class instance containing graph info:
        - mas.edges : list of tuples (i, j)
        - mas.desired_dists : dict {(i, j): desired_distance}
        - mas.leaders : list of leader indices
        - mas.neighbors : dict {i: [neighbor indices]}
    k_f : float
        Formation gain.
    k_p : float
        Tracking gain (used if p_des is provided).
    p_des : dict or None
        Desired positions for leaders {i: np.array([...])}.
    v_des : dict or None
        Desired velocities for leaders {i: np.array([...])}.

    Returns
    -------
    u : dict
        Dictionary of control inputs for leaders {i: np.array([...])}.
    """
    n_agents = mas.N
    n_input = mas.m
    dim = 1 if np.ndim(states) == 1 else states.shape[1]
    if np.ndim(states) == 1:
        states = states.reshape(-1, 1)
    u = np.zeros((n_input, dim))
    for i in mas.leaders:
        u_i = np.zeros((n_input, 1))
        idx_i = np.where(mas.leaders == i)[0][0]
        slc_i_input = slice(idx_i * mas.n_input,
                            (idx_i + 1) * mas.n_input)
        slc_i_state = slice(i * mas.n_state,
                            (i + 1) * mas.n_state)
        u[slc_i_input, :] = -k_p * \
            (states[slc_i_state, :]-desired_pos[idx_i, :].reshape(-1, 1))
    return u


def generate_data(mas, config):
    """
    Simulate the multi-agent system with random control inputs to collect training data.

    Parameters
    ----------
    mas : LeaderFollowerMAS
        The multi-agent system instance.
    config : Config
        Unified dataclass configuration.

    Returns
    -------
    states_array : np.ndarray
        Simulated state trajectories.
    controls_array : np.ndarray
        Applied control inputs.
    cbfs_array : np.ndarray
        Computed CBF values.
    cbfdots_array : np.ndarray
        CBF time derivatives.
    """
    u_ranges = config.agents.u_ranges
    num_states = config.simulation.num_states
    dtSim = config.simulation.dtSim
    TSim = config.simulation.TSim

    def random_control_law(state, t):
        u = []
        for l in mas.leaders:
            idx_l = np.where(l == mas.leaders)[0][0]
            ranges = u_ranges[idx_l]
            for (u_min, u_max) in ranges:
                u.append(np.random.uniform(u_min, u_max))
        return np.array(u)
    # Generate random initial states based on x_ranges
    x_ranges = config.agents.x_ranges
    init_states = []
    for _ in range(num_states):
        state = []
        for i in range(len(x_ranges)):
            dims = x_ranges[i]
            for dim_range in dims:
                state.append(np.random.uniform(*dim_range))
        init_states.append(np.array(state).reshape(1, -1))

    mas.dt = dtSim
    return simulate_mas(mas, init_states, random_control_law, TSim)


def train_jacobian_bounds(mas, states_array, controls_array, cbfdots_array, n_points=300, eps_v=0, save_file: str = "Jbounds.pkl"):
    """Train (estimate) Jacobian bounds."""
    selected_states, selected_controls, selected_hdots = select_points_kmeans(
        states_array.T, controls_array.T, cbfdots_array, n_points)

    # Estimate Jacobian bounds
    Jubounds, Jlbounds = mas.estimate_hdots_jacobian_bounds(
        selected_hdots, selected_states, selected_controls, eps_v=eps_v
    )
    # Saving variables
    with open(save_file, 'wb') as f:
        pickle.dump((Jubounds, Jlbounds), f)  # save as a tuple
    return Jubounds, Jlbounds
