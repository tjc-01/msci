from scipy.stats import norm
import numpy as np
from random import random

class Panel():
    def __init__(self, coords, dim, spread=3, fail_prob=0.001):
        self.coords = coords
        self.dim = dim
        self.spread = spread
        self.fail_prob = fail_prob
        # for each trace, a cell will have an activation 0-1
        self.activations = {}

    def get_activation(self, coords, sigmoid_params=[14,-7]):
        distance = ((self.coords[0] - coords[0])**2 + (self.coords[1] - coords[1])**2)**0.5
        # 0.3989422804014327 is the height of the gaussian, we want a height of 1 for the center
        activation_value = norm.pdf(distance/self.spread, 0, 0.3989422804014327)
        a, b = sigmoid_params
        sigmoid = 1 / (1 + np.e**-(a * activation_value + b))
        response = np.random.choice([0, 1], 1, p=[1 - sigmoid, sigmoid])
        return [activation_value * response[0]] + coords

    def add_noise(self, coords):
        # come up with idea.
        return coords

    # a close by point determined by the detector class has an activation added.
    def add_trace(self, event_name, random_activation=False, coords=None):
        # if not event_name in self.activations.keys():
        #     self.activations[event_name] = []
        if random_activation:
            # change the value from 0.5 to some gaussian func 0-1.0
            pass
            #self.activations[event_name].append(0.5)
        elif (coords is None) or (random() < self.fail_prob):
            pass
           # self.activations[event_name].append(0.0)
        else:
            if not (event_name in self.activations.keys()):
                self.activations[event_name] = []
            ## add noise to coordinate first!!
            noisy_coords = self.add_noise(coords)
            # gaussian amplitude depending on distance from point
            self.activations[event_name].append(self.get_activation(noisy_coords))

    def merge_activations(self, event_name, merge_width):
        if not event_name in self.activations.keys():
            return
        # dynamically merged
        act_list = self.activations[event_name]
        i = 0
        # data act, x, y, t, E
        while True:
            if len(act_list) <= i + 1:
                break
            if (act_list[i+1][3] - act_list[i][3]) <= merge_width:
                act_list[i][0] += act_list[i + 1][0]
                # do properly later act_list[i][0][]
                del act_list[i + 1]
                # if act_list[i+1][0] > act_list[i][0]:
                #     del act_list[i]
                # else:
                #     del act_list[i+1]
            else:
                i += 1




