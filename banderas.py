
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


def intraclusterdist(image, centers, labels, rows, cols):
    dist = 0
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            centroid = centers[labels[label_idx]]
            point = image[i, j, :]
            dist += np.sqrt(np.power(point[0] - centroid[0], 2) + np.power(point[1] - centroid[1], 2) + np.power(
                point[2] - centroid[2], 2))
            label_idx += 1
    return dist

def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

class banderas:

    def __init__(self, flags_image):
        self.flag = flags_image


    def colores(self):
        image = np.array(self.flag, dtype=np.float64) / 255
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        distance = []

        for ncolors in range(1,5):
            image_array_sample = shuffle(image_array, random_state=0)[:10000]
            model = KMeans(n_clusters=ncolors, random_state=0).fit(image_array_sample)
            labels = model.predict(image_array)
            centers = model.cluster_centers_
            dist = intraclusterdist(image, centers, labels, rows, cols)  # se calcula la distancia intra-clusters
            distance.append(dist)  # se adjunta el resultado a la lista de distancias

        distance = np.array(distance)
        minval = np.min(distance[np.nonzero(distance)])
        distances = distance.tolist()
        bestk = distances.index(minval)+1
        print(bestk)
        
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        model = KMeans(n_clusters=bestk, random_state=0).fit(image_array_sample)
        self.labels = model.predict(image_array)
        self.centers = model.cluster_centers_

    def porcentaje(self):

        percentages = []
        for n in range(len(self.centers)):

            cluster = self.labels.tolist()
            cluster = cluster.count(n)
            percentages.append(100*cluster/len(self.labels))

        return percentages

    def orientacion(self):
        None
