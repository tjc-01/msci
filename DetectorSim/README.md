# DetectorSim
Repository dedicated to generating different toy detector geometries for particle reconstruction

## Initialising Detector Object
To create an initial detector object of custom geometry, run the code:

```python
from Detector import *
detector = Detector(dim=(x_dimension, y_dimension))
```

Where x_dimension and y_dimension are the width and height of the panels respectively in 1/10ths of meters. The radius of the detector is set to default length of 27.8 in 1/10ths of meters. 

## Adding tracks

Tracks should be a 2D list/array containing hits, where a hit is formatted as [x_position, y_position, time, energy]. This can then be added to the detector via the command:

```python
detector.add_event(track)
```

all coordinates in this track will also now be labelled with same event number.

## Outputting Data

One can simply output the data in dictionary form via the following:

```python

print(detector.output_data())
#data = detector.output_data()
#print(pd.DataFrame(data))

```

The output data consists of the following information: 

- True x and y position of the registered hits
- The x and y coordinate of the activated panels 
- Gaussian activation of a panel due to a hit 
- The time of the hits 
- Hit energy 

One can proceed to store this data in .pkl, .csv or .json form. You can also display the data in a pandas DataFrame, using the commented code above. 

## Plotting data

One can plot data, after adding a track to the detector. This can be done via:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
detector.plot_panels(fig_ax=(fig, ax))

```