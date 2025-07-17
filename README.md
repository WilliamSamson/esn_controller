# ESN Controller

Echo State Network-based feedforward-feedback controller for nonlinear dynamic systems, with benchmarks for pendulum and robotic arm systems.

## Overview
This project implements an Echo State Network (ESN) controller with PID feedback for controlling nonlinear dynamic systems. It includes benchmarks for a pendulum and a 2-DOF robotic arm, with statistical validation of performance metrics.

## Installation
```bash
pip install -r requirements.txt
python setup.py install

Usage
Run all experiments with:
python run_experiments.py

This generates:

Plots in docs/figures/
Results in benchmarks/results/
Trained models in trained_models/

Project Structure

src/: Core ESN, PID, and RLS modules.
benchmarks/: Benchmark scripts and metrics.
docs/figures/: Visualization outputs.
benchmarks/results/: Statistical results and trial data.
trained_models/: Saved ESN models.

Dependencies

Python >= 3.8
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.4.0

License
MIT License```
