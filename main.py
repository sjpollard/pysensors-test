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

    model = SSPOC(basis=ps.basis.Identity(n_basis_modes=10), l1_penalty=0.005, n_sensors=500)
    model.fit(X_train, y_train)
    print(f'Sensors selected: {len(model.get_selected_sensors())}')

    new_face = np.zeros(faces['images'][0].ravel().shape)
    np.put(new_face, model.get_selected_sensors(), faces['data'][0][model.get_selected_sensors()])
    new_face = np.reshape(new_face, faces['images'][0].shape)
    ts.show(new_face)

    ts.show(model.basis.matrix_representation())

    ts.show(np.reshape(model.basis.matrix_representation().T, (10, 64, 64)))

    #plot_sensor_locations(model.selected_sensors, faces['images'][0].shape)

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