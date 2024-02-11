import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import combinedsim
import preprocessing

import tracksim.consts as consts  # isort: skip # pyright: ignore


def _get_momentum(angles, number):
    output = []
    if len(angles) != number:
        raise Exception("Not enough angles for number of particles.")
    for angle in angles:
        output.append([np.cos(angle) * 300, np.sin(angle) * 300, 0])

    return np.array(output)


def straight_transform(
    hits: NDArray,
    thresh: float,
    bin_size: tuple[float, float],
    line_greediness: tuple[float, float],
    peak_neighbourhood: tuple[int, int],
    origin: NDArray[np.float64] = np.array([0, 0]),
    grad_lim: tuple[float, float] = (-2, 2),
    impact_lim: tuple[float, float] = (-10, 10),
) -> tuple[NDArray, NDArray, list[NDArray]]:
    """Hough transforms straight lines with some specified angular step.

    Args:
        hits (NDArray): An array of hits (preferably with track vertex at the origin).
        thresh (float): The fraction of a a transforms maximum value a bin must meet to
            be considered a real track.
        bin_size (tuple[float, float]): The bin size to be applied to transform space.
        line_greediness (tuple[float, float]): The square in parameter space within
            which a line will consider points that contribute to the accumulator within
            that radius as belonging to it.
        peak_neighbourhood (tuple[int, int]): The size of the neighbourhood in parameter
            space within which a maximum will be considered a peak (as a multiple of
            bin_size).

    Kwargs:
        origin (NDarray): An origin to transform onto before performing the transform.
        grad_lim (tuple[float, float]): Limits search within a fixed gradient
            parameter interval. Defaults to (-10, 10).
        impact_lim (tuple[float, float]): Limits search within a fixed impact parameter
            interval. Defaults to (-10, 10).

    Returns:
        accumulator (NDArray): The unbinned parameter space
            [[angle, impact parameter], ...].
        tracks (NDArray): A set of points in parameter space that correspond to real
            tracks (according to `thresh`).
        line_hits (list[NDArray]): A list of arrays of hits, each element is those hits
            which are linked with a line. Points may be repeated within this structure.
    """
    # transform hits to new origin
    hits = np.array([hit - origin for hit in hits])

    # do transform
    accumulator = []
    for i, hit in enumerate(hits):
        for grad in np.arange(
            grad_lim[0],
            grad_lim[1],
            bin_size[0]
            if abs(hit[0] * bin_size[0]) < abs(bin_size[1])
            else abs(bin_size[1] / hit[0]),
        ):
            accumulator.append([grad, hit[1] - grad * hit[0], i])

    accumulator = np.array(accumulator)

    # bin and filter by threshold
    fig, ax = plt.subplots()
    ax.hist2d(
        *accumulator.T[:-1],
        bins=(
            int((grad_lim[1] - grad_lim[0]) / bin_size[0]),
            int((impact_lim[1] - impact_lim[0]) / bin_size[1]),
        ),
        range=np.array([grad_lim, impact_lim])
    )
    plt.show()
    binned, xedges, yedges = np.histogram2d(
        *accumulator.T[:-1],
        bins=(
            int((grad_lim[1] - grad_lim[0]) / bin_size[0]),
            int((impact_lim[1] - impact_lim[0]) / bin_size[1]),
        ),
        range=np.array([grad_lim, impact_lim])
    )

    # find peaks
    lines = []
    for i, xedge in enumerate(xedges[:-1]):
        for j, yedge in enumerate(yedges[:-1]):
            bin = binned[i, j]
            neighbourhood = binned[
                i - int((peak_neighbourhood[0] - 1) / 2)
                if i - int((peak_neighbourhood[0] - 1) / 2) >= 0
                else 0 : i + int((peak_neighbourhood[0] - 1) / 2) + 1
                if i + int((peak_neighbourhood[0] - 1) / 2) + 1 <= len(binned)
                else len(binned),
                j - int((peak_neighbourhood[0] - 1) / 2)
                if j - int((peak_neighbourhood[0] - 1) / 2) >= 0
                else 0 : j + int((peak_neighbourhood[0] - 1) / 2) + 1
                if j + int((peak_neighbourhood[0] - 1) / 2) + 1 <= len(binned[0])
                else len(binned[0]),
            ]
            if bin == neighbourhood.max() and bin in [
                value
                for value, count in zip(*np.unique(neighbourhood, return_counts=True))
                if count == 1
            ]:
                lines.append(
                    [
                        xedge + 1 / 2 * (xedges[1] - xedges[0]),
                        yedge + 1 / 2 * (yedges[1] - yedges[0]),
                    ]
                )

    # transform back into original coords
    accumulator = np.array(
        [
            [grad, impact + origin[1] - grad * origin[0], i]
            for grad, impact, i in accumulator
        ]
    )
    lines = np.array(
        [[grad, impact + origin[1] - grad * origin[0]] for grad, impact in lines]
    )
    hits = np.array([hit + origin for hit in hits])

    # group hits by line
    line_hits = []
    for line in lines:
        line_hits.append(
            np.array(
                [
                    hits[int(i)]
                    for i in {
                        i
                        for _, _, i in accumulator[
                            (np.abs(accumulator.T[0] - line[0]) < line_greediness[0])
                            & (np.abs(accumulator.T[1] - line[1]) < line_greediness[1])
                        ]
                    }
                ]
            )
        )

    return accumulator, lines, line_hits


angles = np.array([0.3]) + np.pi
data, detector = combinedsim.run_sim(
    {"neut_muon": 1},
    mom_dists={"neut_muon": lambda number: _get_momentum(angles, number)},
)
truth = data["truth"]
hits = preprocessing.geometric_cluster(data["detector"])
hits = hits[["x", "y"]].to_numpy()

accumulator, lines, line_hits = straight_transform(
    hits,
    0.7,
    (np.pi / 3, 20 / 3),
    (0.04, 0.4),
    (3, 3),
    origin=np.array(truth["tracks"][0]["hits"][0][:2]) * 10,  # type: ignore
)

fig, ax = plt.subplots()
detector.plot_panels((fig, ax), events=list(range(5)))
x = np.linspace(-consts.detector["rad"] * 10, consts.detector["rad"] * 10, 100)
for i, line in enumerate(lines):
    plt.plot(x, np.sin(line[0]) / np.cos(line[0]) * x + line[1])
    plt.scatter(*line_hits[i].T)

plt.show()
