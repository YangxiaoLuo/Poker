import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance


class KMeansWithEMD:
    '''
    A Class of K-Means cluster whose distance function is Earth Mover's Distance(EMD).
    Used for the first round.
    '''

    def __init__(self, n_clusters, center, distance_matrix):
        self.n_clusters = n_clusters
        self.center_method = center
        self.distance_matrix = distance_matrix

    def get_rand_center(self, data):
        '''
        Sample k center in data set.
        '''
        centroids = np.random.choice(data, self.n_clusters, replace=False)
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
        elif self.center_method == 'custom':
            centers = np.load('poker_env/ToyPoker/data/toypoker_first_kmeans_centers.npy', allow_pickle=True)
            self.load_centers(centers)
        self.labels_ = np.zeros(n_samples, dtype=np.int)
        self.distances = np.zeros(n_samples)
        self.n_iter_ = 0
        converged = False
        while not converged:
            old_centroids = np.copy(self.cluster_centers_)
            # TODO: parallel
            for i in range(n_samples):
                min_dist, min_index = np.inf, -1
                for j in range(self.n_clusters):
                    dist = KMeansWithEMD.EMD(self.cluster_centers_[j], data[i], self.distance_matrix)
                    if dist < min_dist:
                        min_dist, min_index = dist, j
                        self.labels_[i] = min_index
                        self.distances[i] = min_dist
            for m in range(self.n_clusters):
                potential_list = data[self.labels_ == m]
                # use union as mean of potentials
                self.cluster_centers_[m] = np.asarray(list(set([final_cluster for first_potential in potential_list for final_cluster in first_potential])))
                if self.cluster_centers_[m].size == 0:
                    self.cluster_centers_[m] = np.random.choice(data, 1)[0]
            np.save('poker_env/ToyPoker/data/toypoker_first_kmeans_centers.npy', self.cluster_centers_)
            converged = self.is_converged(old_centroids, self.cluster_centers_)
            self.inertia_ = np.sum(self.distances)
            print("Iteration ", self.n_iter_)
            print("Inertia: ", self.inertia_)
            self.n_iter_ += 1

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

    @classmethod
    def EMD(potential_1, potential_2, distance_matrix):
        '''
        Calculate the EMD distance between two potentials

        Args:
            potential_1(np_array): potential of mean
            potential_2(np_array): potential of point

        Returns:
            total_cost(float): EMD distance between two potentials
        '''
        len_1 = np.size(potential_1)
        len_2 = np.size(potential_2)
        sorted_distance = distance_matrix[np.reshape(potential_1, (-1, 1)), potential_2]
        ordered_clusters = sorted_distance.argsort(axis=0)
        sorted_distance.sort(axis=0)
        mean_remaining = [1/len_1] * len_1
        targets = [1/len_2] * len_2
        done = [False] * len_2
        total_cost = 0
        for i in range(len_1):
            for j in range(len_2):
                if done[j] is True:
                    continue
                mean_cluster = ordered_clusters[i][j]
                amt_remaining = mean_remaining[mean_cluster]
                if amt_remaining == 0:
                    continue
                d = sorted_distance[i][j]
                if amt_remaining < targets[j]:
                    total_cost += amt_remaining * d
                    targets[j] -= amt_remaining
                    amt_remaining = 0
                else:
                    total_cost += targets[j] * d
                    amt_remaining -= targets[j]
                    targets[j] = 0
                    done[j] = True
        return total_cost
