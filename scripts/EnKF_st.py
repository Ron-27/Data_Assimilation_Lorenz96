import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scripts.lorenz96_1d import lorenz96, lorenz96_rk4_step

def EnKF(ensembles, observations, H, R, inflation_factor):
    """
    Ensemble Kalman Filter (EnKF) with inflation.
    """
    Ne = ensembles.shape[1]
    X = ensembles.copy()
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_anomalies = X - X_mean

    # Inflation step
    X = X_mean + inflation_factor * X_anomalies

    # Compute the sample covariance
    P_f = (X_anomalies @ X_anomalies.T) / (Ne - 1)

    # Compute the Kalman gain
    S = H @ P_f @ H.T + R
    K = P_f @ H.T @ np.linalg.inv(S)

    # Update the ensembles with the observations
    Y_perturbed = observations[:, None] + np.random.multivariate_normal(np.zeros(R.shape[0]), R, Ne).T
    for i in range(Ne):
        X[:, i] += K @ (Y_perturbed[:, i] - H @ X[:, i])

    return X

def run_lorenz96_simulation(N, steps, Ne, F_M, dt, nn, nn_o, inflation_factor):
    """
    Run the Lorenz 96 simulation and apply the Ensemble Kalman Filter (EnKF).

    Args:
        N (int): Number of state variables
        steps (int): Number of time steps
        Ne (int): Number of ensemble members
        F_M (float): Forcing term
        dt (float): Time step
        nn (int): time interval for observations
        nn_o (int): Number of observations
        inflation_factor (float): Inflation factor for EnKF
        

    Returns:
        true_states (np.ndarray): Array of true states over time.
        ensemble_states (np.ndarray): Array of ensemble states over time.
        observations (np.ndarray): Array of observations.
    """
    # Initial conditions for ensembles
    x0 = F_M * np.ones(N)
    x0 += np.random.randn(N) * 0.1  # Small perturbation for all states
    ensembles = np.array([x0 + np.random.randn(N) * 0.1 for _ in range(Ne)]).T  # (N, Ne)

    # Create observations
    observation_indices = np.random.choice(N, nn_o, replace=False)  # Randomly select 25 state indices
    observations = np.full((steps, N), np.nan)  # Initialize observations array with NaNs


    # Run the model and store true states and ensemble states
    true_states = []
    ensemble_states = []
    assimilated_states = []

    x = x0.copy()  # (N,)
    for step in range(steps):
        # True state propagation
        x = lorenz96_rk4_step(x, F_M, dt)  # (N,)
        true_states.append(x.copy())  # Append (N,)

        # Ensemble propagation
        for i in range(Ne):
            ensembles[:, i] = lorenz96_rk4_step(ensembles[:, i], F_M, dt)  # ensembles[:, i] is (N,)
        ensemble_states.append(ensembles.copy())  # Append (N, Ne)
        #Observation assimilation
        if step % nn == 0:
            observations[step, observation_indices] = x[observation_indices] + np.random.randn(25) * 4  # N(0, 16) -> std = 4
            Y = observations[step, observation_indices]
            H = np.zeros((len(observation_indices), N))
            H[np.arange(len(observation_indices)), observation_indices] = 1
            R = np.eye(len(observation_indices)) * 16  # Observation error covariance

            ensembles = EnKF(ensembles, Y, H, R, inflation_factor)
            assimilated_states.append(ensembles.copy())
        else:
            assimilated_states.append(ensembles.copy())

    true_states = np.array(true_states)  # (steps, N)
    ensemble_states = np.array(ensemble_states)  # (steps, N, Ne)
    assimilated_states= np.array(assimilated_states)

    return true_states, assimilated_states, observations

def create_animation2(true_states, ensemble_states, observations):
    """
    Create an animation from the Lorenz 96 model outputs using Matplotlib dark mode.

    Args:
        true_states (np.ndarray): Array of true states over time.
        ensemble_states (np.ndarray): Array of ensemble states over time.
        observations (np.ndarray): Array of observations over time.

    Returns:
        anim (FuncAnimation): The Matplotlib animation object.
    """
    # Parameters
    steps, N = true_states.shape  # (steps, N)
    Ne = ensemble_states.shape[2]  # (steps, N, Ne)

    # Calculate square distance between the mean of each ensemble and the true state at each time
    ensemble_means = np.mean(ensemble_states, axis=2)
    square_distances = np.sum((ensemble_means - true_states) ** 2, axis=1)

    # Set up the figure, axis, and plot element
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 14))
    plt.close()

    ax1.set_xlim((0, N - 1))
    ax1.set_ylim((np.min(ensemble_states), np.max(ensemble_states)))

    ax2.set_xlim((0, steps - 1))
    ax2.set_ylim((0, np.max(square_distances)))

    # Increase size of axis labels and tick labels
    label_fontsize = 19
    tick_fontsize = 16

    # Initialize lines
    true_line, = ax1.plot([], [], color='#FE53BB', lw=2, label='Model')  # Neon pink
    ensemble_lines = [ax1.plot([], [], color='#08F7FE', lw=0.5, alpha=0.5)[0] for _ in range(Ne)]  # Neon teal
    observation_scatter = ax1.scatter([], [], color='#FFD700', marker='*', s=100, label='Observations')  # Neon yellow

    distance_line, = ax2.plot([], [], color='#FFFFFF', lw=2, label='Square Distance (ensemble mean from model)')  # White

    # Set axes titles
    ax1.set_xlabel('k', fontsize=label_fontsize)
    ax1.set_ylabel(r'$X_k$', fontsize=label_fontsize)
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.legend(loc='upper right', fontsize=label_fontsize)

    ax2.set_xlabel('Time Step', fontsize=label_fontsize)
    ax2.set_ylabel('Square Distance (ensemble mean from model)', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    #ax2.legend(loc='upper right', fontsize=label_fontsize)

    # Initialization function: plot the background of each frame
    def init():
        true_line.set_data([], [])
        for line in ensemble_lines:
            line.set_data([], [])
        observation_scatter.set_offsets(np.empty((0, 2)))
        distance_line.set_data([], [])
        return [true_line] + ensemble_lines + [observation_scatter] + [distance_line]

    # Animation function. This is called sequentially
    def animate(i):
        x = np.arange(N)
        true_line.set_data(x, true_states[i])
        for line, ensemble in zip(ensemble_lines, ensemble_states[i].T):  # ensemble_states[i] is (N, Ne), ensemble_states[i].T is (Ne, N)
            line.set_data(x, ensemble)

        # Update observations
        obs_indices = np.where(~np.isnan(observations[i]))[0]
        if len(obs_indices) > 0:
            obs_values = observations[i, obs_indices]
            observation_scatter.set_offsets(np.c_[obs_indices, obs_values])
        else:
            observation_scatter.set_offsets(np.empty((0, 2)))

        distance_line.set_data(np.arange(i + 1), square_distances[:i + 1])

        ax1.set_title(f'Time: {i}', fontsize=24)
        return [true_line] + ensemble_lines + [observation_scatter] + [distance_line]

    # Generate the frames list for every 15 time instances
    frames = []
    for i in range(0, steps, 5):
        frames.append(i)
        if np.any(~np.isnan(observations[i])):
            frames.extend([i] * 4)  # Repeat the frame 4 times (1 second pause if interval is 250ms)

    # Call the animator
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=250, blit=True)

    return anim