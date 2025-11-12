# -*- coding: utf-8 -*-
"""

dynamic_system.py

Base class for dynamic systems.

@author: Mirhan Urkmez
Created: 2025-03-08

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


# Base class for dynamic systems
class DynamicSystem:
    """
    Base class for dynamic systems.

    Args:
        n (int): State dimension.
        m (int): Control dimension.
        dt (float): Time step.
        state (np.ndarray, optional): Initial state.
    """

    def __init__(self, n=4, m=1, dt=0.01, state=None):
        """Initialize the dynamic system."""
        self.n = n  # State dimension
        self.m = m  # Control dimension
        self.dt = dt
        self.state = state if state is not None else np.zeros(n)

    def dynamics(self, state, u):
        """This method should be overridden in child classes."""
        raise NotImplementedError(
            "The dynamics method must be implemented by subclasses.")

    def rk4_step(self, state, u):
        """Performs one Runge-Kutta step."""
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u)
        k4 = self.dynamics(state + self.dt * k3, u)
        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, control_law=None, T=5.0, add_noise=False, noise_std=0.01):
        """
        Simulate the system over a time horizon.

        Args:
            control_law (function, optional): Function computing control inputs, default is zero control.
            T (float): Total simulation time.
            add_noise (bool): Whether to add Gaussian noise.
            noise_std (float): Standard deviation of noise.

        Returns:
            tuple: (states, controls) as NumPy arrays.
        """
        # Define a default control law that returns a zero array of appropriate size
        if control_law is None:
            def control_law(state, t): return np.zeros(self.m)

        num_steps = int(T / self.dt)
        # Preallocate memory using self.n
        states = np.zeros((self.n, num_steps + 1))
        # Preallocate memory for control inputs
        controls = np.zeros((self.m, num_steps))
        states[:, 0] = self.state.flatten()  # Set initial state

        for i in range(num_steps):
            t = i * self.dt  # Current time
            u = control_law(states[:, i], t)  # Compute control input

            # Ensure control has correct dimensions
            if np.isscalar(u):
                u = np.array([u])

            controls[:, i] = u  # Store the control input
            next_state = self.rk4_step(states[:, i], u)  # Compute next state

            # Add optional noise
            if add_noise:
                next_state += np.random.normal(0,
                                               noise_std, size=next_state.shape)

            states[:, i + 1] = next_state  # Store new state

        self.state = states[:, -1]  # Update system state
        return states, controls
