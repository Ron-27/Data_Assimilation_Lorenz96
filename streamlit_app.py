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

# Create the animation
anim = create_animation2(true_states, ensemble_states, observations)

# Save the animation to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
    anim.save(tmpfile.name, writer='pillow', fps=7)
    tmpfile_path = tmpfile.name

# Display the animation in Streamlit
st.title("Data Assimilation with Lorenz 96")
st.image(tmpfile_path)

