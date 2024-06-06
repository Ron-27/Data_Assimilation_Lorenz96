# Data Assimilation with Lorenz 96

## Overview
In this project we are performing data assimilation to the 1-dimensional form of the Lorenz 96 system:

$$
\frac{d X_j}  {d t}=\left(X_{j+1}-X_{j-2}\right) X_{j-1}-X_j+F
$$

To go from this:

<div style="text-align: center;">
    <img src="notebooks/gifs/lorenz96_1d.gif" alt="Lorenz 96" width="500"/>
</div>

To this:

<div style="display: flex; justify-content: center;">
    <img src="notebooks/gifs/lorenz96_EnKF.gif" alt="Lorenz 96 EnKF" width="500"/>
</div>

![Lorenz 96 Animation](notebooks/gifs/lorenz96_1d.gif)


![Lorenz 96 Animation](notebooks/gifs/lorenz96_EnKF.gif)

