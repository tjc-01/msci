"""Module containing the TrackGenerator class."""
import json

import numpy as np
import scipy as sp
from numpy.typing import NDArray

import tracksim.cachedrng as cachedrng
import tracksim.consts as consts

import random
import math

def generate_random_point_in_circle(r):

    theta = random.uniform(0, 2 * math.pi)

    radius = math.sqrt(random.uniform(0, 1)) * r

    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    
    return (x, y)

class TrackGenerator(object):
    """Generates idealised particle tracks."""

    def __init__(
        self,
        particle_counts: dict[str, int],
        mom_distris={},
        eloss_fractional=True,
        eloss_percent=0.01,
        mfp=1e-2,
    ):
        """Initialise a TrackGenerator.

        The accepted particle names are:
                    - "muon"

        Args:
            particle_counts (dict): Dictionary of (particle name: number) pairs where
                number is the desired count to be generated.

        Kwargs:
            mom_dists (dict): Dictionary of {"particle name": function(number)} pairs
                where the function generates (n,3) sized arrays from some distribution
                for initial particle momentum.
        """
        self._elossb = eloss_fractional
        self._elossp = eloss_percent
        self._mfp = 1e-2
        self._particle_counts = particle_counts
        self._mom_dists = mom_distris

    def _is_in_detector(self, position: NDArray[np.float64]) -> bool:
        return (
            np.linalg.norm(position[:2]) <= consts.detector["rad"]
            and position[2] > -consts.detector["half_length"]
            and position[2] < consts.detector["half_length"]
        )

    def _get_speed(self, mass: float, energy: float) -> float:
        # using relation for a massive particle E=gamma*m*c^2 to get magnitude of
        # velocity(beta)
        # assume natural units c = 1 (in MeV)
        gamma = energy / mass
        v = 1 - 1 / gamma**2
        return v

    def _track_radius(self, momentum: NDArray[np.float64]) -> float:
        # finds track radius in metres using cyclotron formula.
        # This function definitely gives the correct answer

        # IMPORTANT IF NOT USING MEV AS THE UNIT OF ENERGY THIS SCALE FACTOR NEEDS
        # CHANGE
        length_scale_factor = 10**6 / sp.constants.c
        return length_scale_factor * abs(
            sp.linalg.norm(momentum) / consts.detector["B"]
        )

    def _get_circular_centre(
        self,
        radius: float,
        position: NDArray[np.float64],
        initial_momentum: NDArray[np.float64],
        charge: float,
    ) -> NDArray[np.float64]:
        # computes lorentz force to find direction of the radius of the circle.
        radius_vector = (
            charge
            * np.cross(
                [initial_momentum[0], initial_momentum[1], 0],
                [0, 0, consts.detector["B"]],
            )[:2]
        )
        normalised_radius_vector = radius_vector / np.linalg.norm(radius_vector)
        circle_centre = position[:2] + radius * normalised_radius_vector
        return circle_centre

    # TODO: write get_mfp
    def _get_mfp(self, energy: float, particle_type: str) -> float:
        return self._mfp

    def _get_init_pos(
        self, number: int, mom_s: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x,y = generate_random_point_in_circle(consts.detector["rad"])
        z = np.array([0] * number)
        return np.array([x, y, z]).T

    def _get_init_momentum(
        self, number: int, particle_type: str
    ) -> NDArray[np.float64]:
        if len(self._mom_dists) == 0:
            mom = np.random.normal(1.2e3, 100, size=number)
            angle = np.random.normal(0, 0.08, size=number)
            return np.array([-mom * np.cos(angle), mom * np.sin(angle), [0] * number]).T
        else:
            return self._mom_dists

    def _leave_hit(
        self,
        energy: float,
        position: NDArray[np.float64],
        dist: float,
        crng: cachedrng.CachedRandomGenerator,
        particle_type: str,
    ) -> bool:
        if crng.get_cache() is None:
            crng.generate("exponential", self._get_mfp(energy, particle_type))
        if dist > crng.get_cache():  # type: ignore
            crng.generate("exponential", self._get_mfp(energy, particle_type))
            return (
                True
                and np.linalg.norm(position[:2]) < consts.detector["rad"]
                and position[2] > -consts.detector["half_length"]
                and position[2] < consts.detector["half_length"]
            )
        else:
            return False

    # TODO: write dEdX
    def _denergy_dx(self, energy: float) -> float:
        return self._elossp * energy

    def _generate_track(
        self,
        initial_position: NDArray[np.float64],
        initial_timestamp: float,
        initial_momentum: NDArray[np.float64],
        particle_type: str,
    ) -> tuple[list, float]:
        # take the circle that circumscribes the detector to be the detector radius
        energy = np.sqrt(
            initial_momentum[0] ** 2
            + initial_momentum[1] ** 2
            + np.array(consts.particle["mass"][particle_type]) ** 2
        )
        initial_energy = energy
        v = self._get_speed(consts.particle["mass"][particle_type], energy)
        time_step = 1e-10
        t = time_step
        dist_step = time_step * v * sp.constants.c
        dist_since_hit = dist_step
        current_pos = initial_position
        track_points = []
        crng = cachedrng.CachedRandomGenerator()
        if consts.particle["charge"][particle_type] == 0:
            v_vector = (
                v * initial_momentum / np.linalg.norm(initial_momentum) * sp.constants.c
            )
            while energy > 0 and self._is_in_detector(current_pos) and t < 1e-4:
                if self._leave_hit(
                    energy, current_pos, dist_since_hit, crng, particle_type
                ):
                    current_pos = np.array(
                        [
                            initial_position[0] + v_vector[0] * t,
                            initial_position[1] + v_vector[1] * t,
                            initial_position[2] + v_vector[2] * t,
                        ]
                    )
                    if self._elossb:
                        energy_loss = self._denergy_dx(energy) * dist_since_hit
                    else:
                        energy_loss = self._denergy_dx(initial_energy) * dist_since_hit
                    if energy_loss > energy:
                        energy_loss = energy

                    track_points.append(
                        [
                            current_pos[0],
                            current_pos[1],
                            energy_loss,
                            initial_timestamp + t,
                        ]
                    )

                    energy -= energy_loss
                    dist_since_hit = dist_step

                else:
                    dist_since_hit += dist_step

                t += time_step
            return track_points, energy
        else:
            track_rad = self._track_radius(initial_momentum)
            track_centre = self._get_circular_centre(
                track_rad,
                initial_position,
                initial_momentum,
                consts.particle["charge"][particle_type],
            )
            translated_initial_position = initial_position[:2] - track_centre
            # finds inital phase of the particle's circular track.
            phase = np.arctan2(
                translated_initial_position[1], translated_initial_position[0]
            )
            omega = (
                -np.sign(consts.particle["charge"][particle_type])
                * v
                * sp.constants.c
                / track_rad
            )

            # either waits for particle to leave detector or stops
            # after a certain amount of time to stop infinite loop (though this
            # condition should be replaced in the future with when the particle runs out
            # of energy) for a muon of momemntum c.335MeV/c this gives around a phase
            # rotation of 0.02 pi for every time step.Such a particle has a radius of
            # around 2.23m.

            while energy > 0 and self._is_in_detector(current_pos) and t < 1e-7:
                if self._leave_hit(
                    energy, current_pos, dist_since_hit, crng, particle_type
                ):
                    if self._elossb:
                        energy_loss = self._denergy_dx(energy) * dist_since_hit
                    else:
                        energy_loss = self._denergy_dx(initial_energy) * dist_since_hit
                    if energy_loss > energy:
                        energy_loss = energy

                    if energy_loss > energy:
                        energy_loss = energy

                    track_points.append(
                        [
                            current_pos[0],
                            current_pos[1],
                            energy_loss,
                            initial_timestamp + t,
                        ]
                    )

                    energy -= energy_loss
                    dist_since_hit = dist_step

                else:
                    dist_since_hit += dist_step

                current_pos = np.array(
                    [
                        track_centre[0] + track_rad * np.cos(omega * t + phase),
                        track_centre[1] + track_rad * np.sin(omega * t + phase),
                        current_pos[2]
                        + v
                        * initial_momentum[2]
                        / np.linalg.norm(initial_momentum)
                        * time_step,
                    ]
                )

                t += time_step
            return track_points, energy

    def generate_tracks(self, file=None) -> dict:
        """Generate the series of tracks.

        Kwargs:
            file (str): Path to a file to write output to. Defaults to None.

        Returns:
            tracks: A dictionary containing all the data relevant to the generated
                tracks.
        """
        tracks = []
        track_id = 0
        for particle_type in self._particle_counts:
            count = self._particle_counts[particle_type]
            mom_s = self._get_init_momentum(count, particle_type)
            pos_s = self._get_init_pos(count, mom_s)
            time_s = np.zeros(
                count
            )  # for now all init_time is zero, can be changed later
            for i in range(count):
                hits, energy = self._generate_track(
                    pos_s[i],
                    time_s[i],
                    mom_s[i],
                    particle_type,
                )
                tracks.append(
                    {
                        "id": track_id,
                        "rem energy": energy,
                        "particle": particle_type,
                        "init_mom": mom_s[i].tolist(),
                        "hits": hits,
                    }
                )
                track_id += 1

        output = {"particle_count": self._particle_counts, "tracks": tracks}
        if file is not None:
            with open(file) as f:
                f.writelines(json.dumps(output))
        return output
