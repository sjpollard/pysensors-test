import matplotlib.pyplot as plt
import numpy as np
import pysensors as ps
import torchshow as ts
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pysensors.classification import SSPOC
from PIL import Image, ImageOps


def main():
    # Set seed for reproducibility
    random_state = 1

    display = True

    # Load the data
    X, y, images = load_EPIC('../../EPIC-KITCHENS')

    height, width = images[0].shape

    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    n_basis_modes = 20
    l1_penalty = 0.000005

    #basis = ps.basis.Identity(n_basis_modes=n_basis_modes)
    basis = ps.basis.SVD(n_basis_modes=n_basis_modes)
    #basis = ps.basis.RandomProjection(n_basis_modes=n_basis_modes)

    model = SSPOC(basis=basis, l1_penalty=l1_penalty)
    model.fit(X_train, y_train)
    print(f'Sensors selected: {len(model.get_selected_sensors())}')
    
    if display:
        
        ts.show(images[0])

        plot_sensor_locations(model.selected_sensors, (height, width))

        new_image = np.zeros(height * width)
        np.put(new_image, model.get_selected_sensors(), X[0][model.get_selected_sensors()])
        new_image = np.reshape(new_image, (height, width))
        ts.show(new_image)

        ts.show(model.basis.matrix_representation())

        ts.show(np.reshape(model.basis.matrix_representation().T, (n_basis_modes, height, width)))

    y_pred = model.predict(X_train[:,model.selected_sensors])
    print(f'Train accuracy: {accuracy_score(y_train, y_pred) * 100}%')

    y_pred = model.predict(X_test[:,model.selected_sensors])
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100}%')


def load_EPIC(path):
    annotations = pd.read_csv('EPIC.csv')
    annotations = annotations[annotations['video_id'] == 'P01_01']
    top_3 = annotations['noun_class'].value_counts()[:3].index.tolist()
    annotations = annotations[annotations['noun_class'].isin(top_3)].reset_index(drop=True)
    frames = list(map(str, annotations['stop_frame'].values))
    X = []
    images = []
    for frame in frames:
        zero = '0'
        file_name = f'{(10 - len(frame)) * zero}{frame}'
        x = Image.open(f'{path}/P01/rgb_frames/P01_01/frame_{file_name}.jpg').convert(mode='L')
        X.append(np.ravel(np.asarray(x)/255.0))
        images.append(np.asarray(x)/255.0)
    X = np.stack(X, axis=0)
    images = np.stack(images, axis=0)
    y = np.array(annotations['noun_class'].tolist())
    return X, y, images

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