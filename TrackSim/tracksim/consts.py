"""Contains constants relevant to the particle gun."""
# PARTICLE CONSTANTS

particle = {}
particle["mass"] = {
    "muon": 105.66,
    "neut_muon": 105.66,
    "anti_muon": 105.66,
    "pion+":139.57,
    "pion-":139.57
}  # (MeV / c^2)
particle["charge"] = {"muon": -1, "neut_muon": 0, "anti_muon": 1,"pion+":1, "pion-":-1}  # (e)

# DETECTOR CONSTANTS
detector = {}
detector["rad"] = 2.78  # m
# TODO: find detector length
detector[
    "half_length"
] = 1  # the length of one chamber (i.e. half the detector length) (m)
detector["B"] = 0.5  # (T)
