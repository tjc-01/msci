"""Module that interfaces track and detector simulation to produce final output."""

import json
import pickle
import random
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# fmt: off
# this is a horrible way to do this but I'm really not sure what else to do
# really we should make both submodules python packages and treat them as dependencies
# with poetry - H.B.
sys.path.append("./TrackSim/")
sys.path.append("./DetectorSim/")
sys.path.append("../TrackSim/")
sys.path.append("../DetectorSim/")

from Detector import Detector  # pyright: ignore # noqa
from tracksim.gen import TrackGenerator  # pyright: ignore # noqa

# fmt: on


def run_sim(
    particle_counts: dict[str, int],
    #RETURN mom_dists TO KEYWORD ARGUMENT
    mom_dist,
    detector,
    panel_width=0.5,
    panel_height=0.5,
    path=None,

    eloss_fractional=False,
    eloss_fraction=0.01,
    mfp=1e-2,
) -> tuple[dict, Detector]:
    """Run full simulation and output data to file.

    Args:
        particle_counts (dict): Dictionary of (particle name: number) pairs where
            number is the desired count to be generated.

    Kwargs:
        panel_width (float): Width of the detector panels in metres *10^(-1). Defaults
            to 0.5.
        panel_height (float): Height of the detector panels in metres *10^(-1). Defaults
            to 0.5.
        path (str): Path of the file to save the JSON-encoded output to. Defaults to
            None.
        mom_dists (dict): Dictionary of {"particle name": function(number)} pairs where
            the function generates (n,3) sized arrays from some distribution for initial
            particle momentum. Defaults to an empty dictionary.
        eloss_fractional: If true, the energy lost at each hit is proportional to the
            energy of the particle at each hit. If false it will remove a constant
            fraction of its initial energy each time. Useful to explore algorithms with
            both. Latter is probably more realistic but will never generate a track that
            runs out of energy in the detector.
        eloss_fraction: Set the fraction of energy lost per metre. Set to 1 to get quite
            a short track.
            NOTE: if eloss_fractional is set to false, this sets the energy loss per
                metre as a fraction of the initial energy whereas if it is true it is
                the fractional value of its current energy.
        mfp: set the mean free path to be some constant value. 1e-2 is what has been
            giving reasonable track point density but may want to experiment.

    Returns:
        output (dict): Full output. Dictionary with 2 keys: "truth", and "detector",
            which contain the truth and output data from the simulation. This is the
            dictionary which is saved to path.
        detector (Detector): The detector object used.
    """


    track_generator = TrackGenerator(
        particle_counts,
        detector,
        mom_distris=mom_dist,
        eloss_fractional=eloss_fractional,
        eloss_percent=eloss_fraction,
        mfp=mfp,
    )
 #   detector = Detector(dim=(panel_width, panel_height))
    
    truth = track_generator.generate_tracks()
    output["truth"] = truth

    for track in truth["tracks"]:
        detector.add_event(
            [[x * 10, y * 10, time, energy] for x, y, energy, time in track["hits"]],
            event_name=track["id"],
        )

    output["detector"] = detector.output_data()

    # if path is not None:
    #     with open(path) as f:
    #         f.writelines(json.dumps(output))

    return output



def get_pion_momenta(pion_num):

    pion_moms = np.random.normal(loc = 100, scale = 10 , size = pion_num)
    total_mom = np.random.normal(loc = 1200, scale = 100)
    rem_mom = np.array([np.sqrt(total_mom**2 - np.sum(pion_moms**2)),0,0])
    return (rem_mom.T, np.array([pion_moms, [0]*pion_num, [0]*pion_num]))

    return 

def get_pion_number():
    num = np.random.exponential(scale = 3)
    if num <2:
        return 1
    elif num>5:
        return 5
    else:
        return 3
def get_pion_counts(pion_count, lepton_charge):
    if lepton_charge == 1:
        match pion_count:
            case 1:
                return (0,1)
            case 3:
                return (1,2)
            case 5:
                return (2,3)

    if lepton_charge == -1:
        match pion_count:
            case 1:
                return (1,0)
            case 3:
                return (2,1)
            case 5:
                return (3,2)

events = []
events_truth = []

for i in tqdm(range(1)):
    vertex_number = np.random.poisson(3)
    pion_counts = []
    for i in range(vertex_number):
        pion_counts.append(get_pion_number())
    print("v", vertex_number)
    print("pion counts")
    muon_counts = np.random.randint(2, size=vertex_number)
    print("muon_counts", muon_counts)
    detector = Detector(dim=(0.5, 0.5))
    
    for i in range(vertex_number):
        if muon_counts[i] == 1:
            lepton_charge = 1
        else:
            lepton_charge = -1
        print("q", lepton_charge)
        pions = get_pion_counts(pion_counts[i], lepton_charge)
        print("pion nums", pions)
        (muon_mom, pion_moms) = get_pion_momenta(pion_counts[i])
        print("pion moms", pion_moms)
        print(pions)
        moms = {"muon":muon_mom, "anti_muon":muon_mom, "pion+":pion_moms[:,:pions[0]].T, "pion-":pion_moms[:,pions[0]:].T}
        print(moms)
        output = run_sim(
            {"muon": (lepton_charge == -1), "anti_muon": int(lepton_charge == +1), "pion+":pions[0], "pion-":pions[1]}, 
            moms,
            detector,
        )
    output = {}
    output["detector"] = detector.output_data()
    output["detector"]["energy"] = (
        output["detector"]["energy"] * output["detector"]["activation"]
    )
    events.append(
        output["detector"][["panel_x", "panel_y", "energy", "time"]].to_dict()
    )
    output["detector"] = output["detector"].to_dict()
    events_truth.append(output)
    fig, ax = plt.subplots()
    detector.plot_panels((fig, ax), events=range(10))
    plt.show()

with open("piontest.json", "w") as f:
    f.writelines(json.dumps(events))

with open("piontesttruth.json", "w") as f:
    f.writelines(json.dumps(events_truth))
