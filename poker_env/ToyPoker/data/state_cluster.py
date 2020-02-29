import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance


class KMeansWithEMD:
    '''
    A Class of K-Means cluster whose distance function is Earth Mover's Distance(EMD).
    Used for the first round.
    '''

    def __init__(self, n_clusters=200, center='rand'):
        self.n_clusters = n_clusters
        self.center_method = center

    def get_rand_center(self, data):
        '''
        Generate k center within the range of data set.
        '''
        n_features = data.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        for i in range(n_features):
            d_min, d_max = np.min(data[:, i]), np.max(data[:, i])
            centroids[:, i] = np.random.rand(self.n_clusters) * (d_max - d_min) + d_min
        return centroids

    def get_euclid_center(self, data):
        '''
        Generate n cluster_center through sklearn.KMeans.
        '''
        k_means = KMeans(self.n_clusters)
        k_means.fit(data)
        return k_means.cluster_centers_

    def is_converged(self, centroids1, centroids2):
        set1 = set([tuple(c) for c in centroids1])
        set2 = set([tuple(c) for c in centroids2])
        return (set1 == set2)

    def fit(self, data):
        n_samples = data.shape[0]
        if self.center_method == 'rand':
            self.cluster_centers_ = self.get_rand_center(data)
        elif self.center_method == 'euclid':
            self.cluster_centers_ = self.get_euclid_center(data)
        self.labels_ = np.zeros(n_samples, dtype=np.int)
        self.distances = np.zeros(n_samples)
        self.n_iter_ = 0

        converged = False
        while not converged:
            self.n_iter_ += 1
            old_centroids = np.copy(self.cluster_centers_)
            for i in range(n_samples):
                min_dist, min_index = np.inf, -1
                for j in range(self.n_clusters):
                    dist = wasserstein_distance(data[i], self.cluster_centers_[j])
                    if dist < min_dist:
                        min_dist, min_index = dist, j
                        self.labels_[i] = min_index
                        self.distances[i] = min_dist
            for m in range(self.n_clusters):
                self.cluster_centers_[m] = np.mean(data[self.labels_ == m], axis=0)
            converged = self.is_converged(old_centroids, self.cluster_centers_)

        self.inertia_ = np.sum(self.distances)

    def load_centers(self, cluster_centers):
        '''
        Set the cluster center manually.

        Args:
            cluster_centers (np.ndarray): shape is (n_clusters, n_features)
        '''
        self.cluster_centers_ = cluster_centers

    def load_default_centers(self):
        '''
        Load the default 200-centers.
        '''
        data_path = 'poker_env/Toypoker/data/toypoker_first_kmeans_centers.npy'
        precompute_centers = np.load(data_path)
        self.load_centers(precompute_centers)

    @classmethod
    def default_training(self, data_path='poker_env/Toypoker/data/toypoker_first_ehs_vector.csv',
                         save_path='poker_env/Toypoker/data/toypoker_first_kmeans_centers.npy'):
        '''
        Start default training and save the result.
        '''
        first_data = pd.read_csv(data_path, index_col=None)
        train_data = first_data[first_data.columns[2:]].values
        emd_kmeans = KMeansWithEMD(n_clusters=200, center='euclid')
        emd_kmeans.fit(train_data)
        np.save(save_path, emd_kmeans.cluster_centers_)

    def predict(self, data):
        '''
        Predict the abstract clusters of the data.

        Args:
            data (np.ndarray): shape is (n_samples, n_features)

        Returns:
            (np.ndarray): Index of the cluster each sample belongs to. shape is (n_samples, 1)
        '''
        n_samples = data.shape[0]
        result_labels = np.zeros((n_samples, 1), dtype=np.int)
        for i in range(n_samples):
            min_dist, min_index = np.inf, -1
            for j in range(self.n_clusters):
                dist = wasserstein_distance(data[i], self.cluster_centers_[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    result_labels[i] = min_index
        return result_labels
