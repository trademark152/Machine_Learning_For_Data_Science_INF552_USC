import numpy as np
# from data_loader import toy_dataset, load_digits
from kmeans import KMeans, KMeansClassifier, get_k_means_plus_plus_center_indices as k_plus, get_lloyd_k_means as k_vanilla, transform_image
from utils import Figure
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def toy_dataset(total_clusters=2, sample_per_cluster=50):
    # TODO: add sample size ease
    np.random.seed(int(time.time()))
    N = total_clusters * sample_per_cluster
    y = np.zeros(N)
    np.random.seed(43)
    x = np.random.standard_normal(size=(N, 2))
    for i in range(total_clusters):
        theta = 2 * np.pi * i / total_clusters
        x[i * sample_per_cluster:(i + 1) * sample_per_cluster] = x[
                                                                 i * sample_per_cluster:(i + 1) * sample_per_cluster] + \
                                                                 (total_clusters * np.cos(theta),
                                                                  total_clusters * np.sin(theta))
        y[i * sample_per_cluster:(i + 1) * sample_per_cluster] = i

    return x, y

""" perform k means on toy dataset, invoke each type of centroid function"""
def kmeans_toy():
    print("[+] K-Means on Toy Dataset")

    print("[+] K-Means Vanilla")
    kmeans_builder(k_vanilla)
    print()

    print("[+] K-Means Plus Plus")
    kmeans_builder(k_plus)
    print()

""" builder for kmeans: input is the type of centroid function use"""
def kmeans_builder(centroid_func):
    samples_per_cluster = 50
    n_cluster = 9

    x, y = toy_dataset(n_cluster, samples_per_cluster)
    # print("x: ", x)
    # print("y: ", y)

    # plot the scatter plot with color coded by cluster index
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=y)
    fig.savefig('plots/toy_dataset_real_labels.png')

    fig.ax.scatter(x[:, 0], x[:, 1])
    fig.savefig('plots/toy_dataset.png')

    # create a class kmeans
    k_means = KMeans(n_cluster=n_cluster, max_iter=100, e=1e-8)

    # fit the kmeans to data x using centroid_func to initialize
    centroids, membership, i = k_means.fit(x, centroid_func)

    assert centroids.shape == (n_cluster, 2), \
        ('centroids for toy dataset should be numpy array of size {} X 2'
            .format(n_cluster))

    assert membership.shape == (samples_per_cluster * n_cluster,), \
        'membership for toy dataset should be a vector of size {}'.format(len(membership))

    assert type(i) == int and i > 0,  \
        'Number of updates for toy datasets should be integer and positive'

    print('[success] : kmeans clustering done on toy dataset')
    print('Toy dataset K means clustering converged in {} steps'.format(i))

    # plot toy dataset labelling using OUR method
    fig = Figure()
    fig.ax.scatter(x[:, 0], x[:, 1], c=membership)
    fig.ax.scatter(centroids[:, 0], centroids[:, 1], c='red')
    # plt.show()
    fig.savefig('plots/toy_dataset_predicted_labels.png')


if __name__ == '__main__':
    kmeans_toy()