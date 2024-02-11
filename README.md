# CombinedSim
Combination of both the track and detector simulation.

## Installation
Uses [python-poetry](https://python-poetry.org/) for dependency management. To install a venv:
1. Install poetry (`pip install poetry`).
2. `poetry install` while in the top-level directory.
3. `poetry shell` opens a subshell inside the virtual environment.

## Usage
Everything important happens within the `combinedsim` module. It can be used as follows:
```
import combinedsim
combinedsim.run_sim({"muon": 5}, 0.5, 0.5) # e.g. runs a simulation of 5 muon tracks with 0.05x0.05 m panels, outputs the data, and produces plots
```
