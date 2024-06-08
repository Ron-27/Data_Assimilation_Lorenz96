import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import sys
import os
import tempfile
# Add the project root directory to the Python path
sys.path.append(os.path.abspath('..'))


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scripts.lorenz96_1d import lorenz96, lorenz96_rk4_step
from scripts.EnKF_st import EnKF, run_lorenz96_simulation, create_animation2


Ne = st.sidebar.slider("Number of Ensembles", min_value=10, max_value=50, value=20, step=10)
infla = st.sidebar.slider("Inflation Factor", min_value=1.0, max_value=1.3, value=1.01, step=0.01)



# Example usage in Streamlit
N = 40
steps = 300
#Ne = 20
F_M = 8.0
dt = 0.01
nn = 15
nn_o = 25
inflation_factor = 1.02

true_states, ensemble_states, observations = run_lorenz96_simulation(
        N, steps, Ne, F_M, dt, nn, nn_o, infla)

# Function to create data for animation
def prepare_data(true_states, ensemble_states, observations):
    steps, N = true_states.shape
    Ne = ensemble_states.shape[2]

    # Prepare data for Plotly
    data = []
    for i in range(steps):
        for k in range(Ne):
            for j in range(N):
                data.append({
                    'x': j,
                    'y': ensemble_states[i, j, k],
                    'line_id': f'Ensemble {k}',
                    'frame_id': i,
                    'type': 'Ensemble'
                })
        for j in range(N):
            data.append({
                'x': j,
                'y': true_states[i, j],
                'line_id': 'True State',
                'frame_id': i,
                'type': 'True State'
            })
        obs_indices = np.where(~np.isnan(observations[i]))[0]
        if len(obs_indices) > 0:
            obs_values = observations[i, obs_indices]
            for j, value in zip(obs_indices, obs_values):
                data.append({
                    'x': j,
                    'y': value,
                    'line_id': 'Observations',
                    'frame_id': i,
                    'type': 'Observations'
                })

    df = pd.DataFrame(data)
    return df

# Function to create animation using Plotly
def create_animation2(true_states, ensemble_states, observations):
    df = prepare_data(true_states, ensemble_states, observations)

    # Plotting with Plotly Express
    fig = px.line(df[df['type'] != 'Observations'], x='x', y='y', animation_frame='frame_id', animation_group='line_id', 
                  line_group='line_id', color='line_id')

    # Customize colors and line settings
    fig.for_each_trace(lambda trace: trace.update(line=dict(color='#08F7FE', width=1), opacity=0.5) if 'Ensemble' in trace.name else trace.update(line=dict(color='#FE53BB', width=2)))
    
    # Add observations as scatter points only when they are not NaN
    scatter_frames = []
    for i in df['frame_id'].unique():
        frame_data = df[(df['frame_id'] == i) & (df['type'] == 'Observations')]
        if not frame_data.empty:
            scatter_frames.append(frame_data)

    scatter_traces = [px.scatter(frame, x='x', y='y').data[0] for frame in scatter_frames]
    for trace in scatter_traces:
        trace.update(marker=dict(color='#FFD700', symbol='star', size=10), showlegend=False, name='Observations')
        fig.add_trace(trace)

    fig.update_layout(
        title="Lorenz 96 Model Data Assimilation",
        xaxis_title="State Variable",
        yaxis_title="Value",
        showlegend=False,
        height=700
    )

    return fig



# Create the animation
fig = create_animation2(true_states, ensemble_states, observations)

# Display the animation in Streamlit
st.title("Lorenz 96 Model Data Assimilation Animation")
st.plotly_chart(fig)