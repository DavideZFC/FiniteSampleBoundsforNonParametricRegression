# Finite Sample Bounds for Non-Parametric Regression: Optimal Sample Efficiency and Space Complexity

This repository contains the official Python implementation for the numerical simulations presented in **Appendix H ("Numerical Simulations: Extensions and Ablation Studies")** of the paper:

> **Finite Sample Bounds for Non-Parametric Regression: Optimal Sample Efficiency and Space Complexity** > *Under review as a submission to TMLR* > Preprint: [arXiv:2412.14744](https://arxiv.org/abs/2412.14744)

## Overview

The codebase provides an empirical validation of the **DUPA (Derivative-Uniform Parametric Approximation)** algorithm. It compares DUPA's minimax-optimal uniform convergence rates under supremum norm ($\|\cdot\|_{\infty}$) against standard, un-perturbed linear regression baselines, highlighting how the "Kernel Perturbation Trick" successfully eliminates misspecification bias.

## Repository Structure

The repository is divided into two folders: functions and classes. The former contains basis functions to run the experiment. Specifically,
* `dirichlet/poussin.py`: Define the positive and negative parts of the corresponding kernels, to be given in input to the Kernel class.
* `fourier_features.py`: Defines the multivariate Fourier feature map.
* `optimal_design.py`: Functions realted to optimal design (KW algorithm to be used in passive design_query.py.
* `three_d_plotter.py`: Functions to make 3D plots of the results

classes contains the following core modules:
* `curves.py`: Defines the target function
* `passive_design_query.py`: Defines a passive design based on optimal design for a feature map.
* `kernel.py`: Defines the `Kernel` class, implementing the sampling strategy from the positive and negative parts of the kernel given in input.
* `environment.py`: Contains `FitterEnv`, which simulates the black-box data-generating environment embedding the true smooth target curve $f(x)$ and generates sub-Gaussian response noise.
* `experiment.py`: Contains the `Experiment` class that orchestrates the simulation pipelines, running both the standard baseline experiments and the Dupa kernel-perturbed experiments, and reporting RMSE and Maximum Absolute Error.


## Getting Started

Run `main.py` for a simple regression experiment. This script configures the simulation environment with a two-dimensional Gaussian target function and performs 
a comparative analysis between different perturbation kernels (Dirichlet and De la Vallée-Poussin). It evaluates the model's performance by calculating its error
over different samples and random seeds to empirically reproduce the theoretical bounds of the paper. Relevant parameters like degree of the Fourier features, number of samples and number of seeds are directly defined in the script.

### Prerequisites

Ensure you have Python 3.8+ installed. The project relies on standard libraries: `numpy`, `matplotlib` and `json`.
