import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import pysensors as ps
from pysensors.classification import SSPOC

import torchshow as ts

def main():
    # Set seed for reproducibility
    random_state = 0
    
    # Load the data
    faces = datasets.fetch_olivetti_faces()
    X = faces['data']
    y = faces['target']

    ts.show(faces['images'][0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    model = SSPOC(l1_penalty=0.01)
    model.fit(X_train, y_train)
    print(model.get_selected_sensors())
    plot_sensor_locations(model.selected_sensors, faces['images'][0].shape)

# Plot sensor locations
def plot_sensor_locations(sensors, shape, ax=None):
    img = np.zeros(shape[0] * shape[1])
    img[sensors] = 1

    if ax is None:
        plt.imshow(img.reshape(shape[0], shape[1]), cmap=plt.cm.binary)
        plt.xticks([])
        plt.yticks([])
        plt.title('Learned sensor locations')
    else:
        ax.imshow(img.reshape(shape[0], shape[1]), cmap=plt.cm.binary)
        ax.set(xticks=[], yticks=[], title='Learned sensor locations')

    plt.show()

if __name__ == "__main__":
    main()