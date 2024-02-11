# TrackSim
Simulation of a simplified particle gun, track generator. 

## Installation
Uses [python-poetry](https://python-poetry.org/) for dependency management. To install a venv:
1. Install poetry (`pip install poetry`).
2. `poetry install` while in the top-level directory.
3. `poetry shell` opens a subshell inside the virtual environment.

## Usage
With the exception of plotting functions, all functionality is contained within the `tracksim.gen.TrackGenerator` class, it can be used as follows:
```
import tracksim.gen as tsg
tg = tsg.TrackGenerator({"muon": 5}, {"muon": some_callable_momentum_dist}) # instantiate the TrackGenerator with a dictionary of particle types and counts, and a dictionary of particle types and momentum distributions (N.B. this can be an empty dictionary, in which case the momenta will be drawn from a predefined normal cone) 
tracks = tg.generate_tracks(file="/some/file/path") # generate track list (where file is an optional file to write tracks to)
```
Full documentation is contained within the docstrings. Use `help(object_or_method)` to obtain type signatures and details.

For plotting tracks you can do:
```
import tracksim.gen as tsg
import tracksim.plotting as tsp

tg = tsg.TrackGenerator(...)
tg.generate_tracks()
tsp.plot_tracks(tracks)
```
