import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import combinedsim as cs


def geometric_cluster(hits, t_step=2e-9, cut_off=0.8, dist=0.6, merge=True):
    hits = hits.copy()
    bins = np.arange(hits.time.min(), hits.time.max(), t_step)
    group_by = hits.groupby(pd.cut(hits.time, bins))
    # group = group_by.get_group(list(group_by.groups.keys())[0])
    # look for 3*3
    data_dict = {"x": [], "y": [], "error": [], "time": []}
    for _, group in group_by:
        group = group.sort_values(["activation"], ascending=False)
        bool_arr = group.activation > cut_off
        activation_data = zip(
            group.activation[bool_arr],
            group.panel_x[bool_arr],
            group.panel_y[bool_arr],
        )
        nn_groups = {}
        activations_checked = []
        for activation, x, y in activation_data:
            if activation in activations_checked:
                continue
            nn_group = group[
                (
                    (group.panel_x >= x - dist)
                    & (group.panel_x <= x + dist)
                    & (group.panel_y >= y - dist)
                    & (group.panel_y <= y + dist)
                )
            ]
            nn_group_act_max = nn_group.activation.max()
            if nn_group_act_max == activation:
                activations_checked += list(nn_group.activation)
                nn_groups[activation] = nn_group
            else:
                activations_checked += list(nn_group.activation)
                if merge:
                    for group_name, group in nn_groups.items():
                        if (group.activation == nn_group_act_max).sum():
                            nn_groups[group_name] = pd.merge(
                                group, nn_group, how="outer"
                            )
                else:
                    nn_groups[activation] = nn_group

        for nn_group in nn_groups.values():
            central_point = (
                (nn_group.activation.dot(nn_group[["panel_x", "panel_y"]]))
            ) / nn_group.activation.sum()
            error = 1 / np.sqrt(len(nn_group))
            data_dict["x"].append(central_point[0])
            data_dict["y"].append(central_point[1])
            data_dict["error"].append(error)
            data_dict["time"].append(nn_group["time"].mean())

    return pd.DataFrame(data_dict)
