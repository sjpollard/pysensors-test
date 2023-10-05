import matplotlib.pyplot as plt
import numpy as np
import pysensors as ps
import torchshow as ts
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pysensors.classification import SSPOC
from PIL import Image




def main():
    # Set seed for reproducibility
    random_state = 0

    display = False

    # Load the data
    faces = datasets.fetch_olivetti_faces()
    X = faces['data']
    y = faces['target']

    height, width = faces['images'][0].shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    n_basis_modes = 100
    l1_penalty = 0.2

    #basis = ps.basis.Identity(n_basis_modes=n_basis_modes)
    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)
    #basis = ps.basis.RandomProjection(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, l1_penalty=l1_penalty)
    model.fit(X_train, y_train)
    print(f'Sensors selected: {len(model.get_selected_sensors())}')
    
    if display:
        
        ts.show(faces['images'][0])

        plot_sensor_locations(model.selected_sensors, (height, width))

        new_face = np.zeros(height * width)
        np.put(new_face, model.get_selected_sensors(), faces['data'][0][model.get_selected_sensors()])
        new_face = np.reshape(new_face, (height, width))
        ts.show(new_face)

        ts.show(model.basis.matrix_representation())

        ts.show(np.reshape(model.basis.matrix_representation().T, (n_basis_modes, height, width)))

    y_pred = model.predict(X_train[:,model.selected_sensors])
    print(f'Train accuracy: {accuracy_score(y_train, y_pred) * 100}%')

    y_pred = model.predict(X_test[:,model.selected_sensors])
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100}%')


def load_EPIC(path):
    annotations = pd.read_csv('EPIC.csv')
    annotations = annotations[annotations['video_id'] == 'P01_01']
    print(annotations['noun_class'].value_counts())
    #ts.show(Image.open(f'{path}/P01/rgb_frames/P01_01/frame_0000000001.jpg'))

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
    load_EPIC('../../EPIC-KITCHENS')