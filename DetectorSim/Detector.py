import numpy as np
from Panel import Panel
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.widgets import Slider
import random
import pandas as pd
from scipy.stats import norm


# dim = (cell_height, cell_width)
# radius=27.8, dim=(3.5,3.5)

class Detector():
    def __init__(self, radius=27.8, dim=(0.5, 0.5),
                 spread=0, ran_prob=0.0001, fail_prob=0.01):
        self.radius = radius
        self.dim = dim
        if spread == 0:
            self.spread = dim[0] * 1.5
        else:
            self.spread = spread
        self.dim = dim
        self.ran_prob = ran_prob
        self.fail_prob = fail_prob
        self.natural_data = []
        self.generate()

    def x(self, y):
        return np.sqrt(self.radius ** 2 - y ** 2)

    def add_Panel(self, coords):
        self.panel_dict[coords] = Panel(coords=coords, dim=self.dim,
                                        spread=self.spread, fail_prob=self.fail_prob)

    def plot_panels(self, fig_ax=None, events=[1], time_range=[0, 10000], show_truth=False):
        alpha = 1
        normaliser = self.output_data()['activation'].max()
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        for panel in self.panel_dict.values():
            current_max = 0
            current_time = 0
            for event in events:
                if event not in panel.activations.keys():
                    continue
                act_max = max([coord[0] for coord in panel.activations[event]])
                time = [coord[3] for coord in panel.activations[event]]

                if act_max > current_max:
                    current_max = act_max
                    current_time = time
            alpha = current_max / normaliser
            fill = False
            if alpha > 0:
                for t in time:
                    if (t <= time_range[1]) and (t >= time_range[0]):
                        fill = True
                        break

            ax.add_patch(Rectangle(xy=(panel.coords[0] - self.dim[1] / 2, panel.coords[1] - self.dim[0] / 2),
                                   width=self.dim[1],
                                   height=self.dim[0],
                                   edgecolor='red',
                                   facecolor=(0, 0, 1, alpha),
                                   fill=fill,
                                   lw=self.dim[0] / 5
                                   ))
        if show_truth:
            ax.plot(np.array(self.natural_data)[:, 0], np.array(self.natural_data)[:, 1], '.',
                    alpha=np.array(self.natural_data)[:, 2] / 20)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim([-self.radius, self.radius])
        ax.set_ylim([-self.radius, self.radius])

    def generate(self):
        self.panel_dict = {}
        cell_height = self.dim[0]
        cell_width = self.dim[1]
        height = cell_height
        while height < self.radius:
            no_cells_wide = int(((self.radius ** 2 - height ** 2) ** 0.5) / cell_width)
            for i in range(no_cells_wide):
                y = height - cell_height / 2
                x = (i * cell_width) + cell_width / 2
                self.add_Panel((x, y))
                self.add_Panel((x, -y))
                self.add_Panel((-x, y))
                self.add_Panel((-x, -y))
            height += cell_height

    def get_panels(self):
        return self.panel_dict

    def find_panel(self, coords, event_name):
        x = coords[0]
        y = coords[1]
        height = self.dim[0]
        width = self.dim[1]
        if x > self.radius or y > self.radius:
            raise Exception("Point not inside detector.")
        else:
            if x != 0 and y != 0:
                panel_x = (int(x / width) * width) + (np.sign(x) * width / 2)
                panel_y = (int(y / height) * height) + (np.sign(y) * height / 2)
            elif x == 0 and y != 0:
                sign = random.choice([-1, 1])
                panel_x = (int(x / width) * width) + (sign * width / 2)
                panel_y = (int(y / height) * height) + (np.sign(y) * height / 2)
            elif y == 0 and x != 0:
                sign = random.choice([-1, 1])
                panel_x = (int(x / width) * width) + (np.sign(x) * width / 2)
                panel_y = (int(y / height) * height) + (sign * height / 2)
            elif x == 0 and y == 0:
                sign = random.choice([-1, 1])
                panel_x = (int(x / width) * width) + (sign * width / 2)
                panel_y = (int(y / height) * height) + (sign * height / 2)

            signals = [[1, 0], [-1, 0],
                       [0, -1], [0, 1],
                       [1, 1], [1, -1],
                       [-1, 1], [-1, -1]]

            if (panel_x, panel_y) in self.panel_dict.keys():
                self.panel_dict[(panel_x, panel_y)].add_trace(
                    event_name, coords=coords)

            for signal in signals:
                if (panel_x + (signal[0] * width), panel_y + (signal[1] * height)) in self.panel_dict.keys():
                    self.panel_dict[(panel_x + (signal[0] * width),
                                     panel_y + (signal[1] * height))].add_trace(
                        event_name, coords=coords)

    # incase tracks have overlapping time.
    # take the mean
    def merge_tracks(self, df, merge_width=1e-9):
        return (
            df.groupby(['panel_x', 'panel_y', 'time']).agg(
                event=pd.NamedAgg(column="event", aggfunc="sum"),
                activation=pd.NamedAgg(column="activation", aggfunc="sum"),
                energy=pd.NamedAgg(column="energy", aggfunc="sum"),
                real_x=pd.NamedAgg(column="real_x", aggfunc="mean"),
                real_y=pd.NamedAgg(column="real_y", aggfunc="mean")
            ).reset_index())

    def output_data(self, merge=True, merge_width=1e-9):
        data_structure = {'event': [], 'panel_x': [], 'panel_y': [], 'time': [],
                          'activation': [], 'energy': [], 'real_x': [], 'real_y': []}
        for (panel_x, panel_y), panel in self.panel_dict.items():
            for event_name, event_activation in panel.activations.items():
                data_structure['event'] += [event_name] * len(event_activation)
                data_structure['panel_x'] += [panel_x] * len(event_activation)
                data_structure['panel_y'] += [panel_y] * len(event_activation)
                for data in event_activation:
                    for d, label in zip(data, ['activation', 'real_x', 'real_y',
                                               'time', 'energy']):
                        data_structure[label].append(d)
        output_df = pd.DataFrame(data_structure)
        output_df.event = output_df.event.astype('str')
        output_df = output_df[output_df.activation != 0]
        if merge:
            output_df = self.merge_tracks(output_df, merge_width=merge_width)
        return output_df

    def apply_kernel(self, mat):
        return

    def add_event(self, event_coords, event_name=1, merge_width=3e-9):
        for coords in event_coords:
            self.find_panel(coords, event_name)
            self.natural_data.append([coords[0], coords[1], coords[2]])
        self.apply_kernel([])
        # merge relevant points
        for panel in self.panel_dict.values():
            panel.merge_activations(event_name, merge_width)

    def add_events(self, event_coord_list, event_names=[1], merge_width=3e-9):
        for event_coords, event_name in zip(event_coord_list, event_names, merge_width):
            self.add_event(event_coords, event_name)

        # find nearby pane;s to coords and add activations


if __name__ == "__main__":
    detector = Detector(dim=(0.5, 0.5))
    fig, ax = plt.subplots()
    # track = [[-5, -6.1, 1, 1e-3], [12.5, 2.0, 2, 1e-3], [-6.0, 5.3, 2, 1e-3], [8.11 ,8.12, 4, 1e-3]]#, (12.31, 12.12, 5), (-2.12, -1.73, 6)
    track_2 = []
    for i in range(-25, 25):
        track_2.append([-6.7 + i / 2, 5.3 + i / 2, 25 + i, 1e-3])
        track_2.append([-3.0 - i / 2, 9 + i / 2, 25 + i, 1e-3])

    detector.add_event(track_2)
    detector.plot_panels(fig_ax=(fig, ax))
    data = detector.output_data()
    print(pd.DataFrame(data))

#     d[hit[2]].append([hit[0], hit[1]])
# fig, ax = plt.subplots()
# def update(val):
#     detector = Detector()
#     #fig, ax = plt.subplots()
#     detector.plot_panels(ax=ax, fig_ax=fig)
#     t = time.val
#     for hit in d[t]:
#         detector.find_panel(hit)

#     detector.plot_panels(ax=ax, fig_ax=fig)
# #def test():
# detector = Detector()
# #detector.plot_panels(ax=ax, fig_ax=fig)
# t = np.linspace(1, 10, 10)
# axtime = plt.axes([0.25, 0.2, 0.65, 0.03])
# time = Slider(axtime, 'time', 0, 10, 1, valstep=[0,1,2,3,4,5,6,7,8,9,10])
# track = [(-5, -6.1, 1), (3.0, 2.0, 2), (-6.0, 4.0, 2)]#, (8.11 ,8.12, 4), (12.31, 12.12, 5), (-2.12, -1.73, 6))
# d = {}
# for hit in track:
#     if hit[2] not in d:
#         d[hit[2]] = []

#     d[hit[2]].append([hit[0], hit[1]])


# time.on_changed(update)
# print('here')
# plt.show()
# for keys, items in d.items():
#     for hit in items:
#         detector.find_panel(hit)
# for hit in track:
#     detector.find_panel(hit)
# detector.plot_panels()
# print(detector.get_panels()[list(detector.get_panels().keys())[0]])

# test()
'''coordinate given in form of (x, y, t)'''