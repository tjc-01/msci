"""Test to assess the average distance traversed between hits for a track."""

import numpy as np

import tracksim.gen as tsg

tg = tsg.TrackGenerator({"muon": 10})
tracks = tg.generate_tracks()

for track in tracks["tracks"]:
    dists = []
    for hit1, hit2 in zip(track["hits"], track["hits"][1:]):
        dists.append(np.sqrt((hit2[0] - hit1[0]) ** 2 + (hit2[1] - hit1[1]) ** 2))
    print(f"ID: {track['id']}\nmom: {track['init_mom']}\navg_dist: {np.average(dists)}")
