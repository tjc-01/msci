"""Generates plots for testing."""
import tracksim.gen as tsg
import tracksim.plotting as tsp

tg = tsg.TrackGenerator({"muon": 5})
tsp.plot_tracks(tg.generate_tracks())
