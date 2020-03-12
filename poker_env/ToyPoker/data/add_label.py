import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from poker_env.ToyPoker.data.kmeans_emd import KMeansWithEMD


def final_kmeans(n_clusters):
    '''
    Add labels in toy_poker_final_ehs.csv

    Args:
        n_clusters(int): number of clusters for kmeans
    '''
    table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', index_col=None, low_memory=False)
    if os.path.isfile('poker_env/ToyPoker/data/toypoker_final_kmeans_centers.npy') is False:
        train_data = (table['ehs'].to_numpy()).reshape(-1, 1)
        k_means = KMeans(n_clusters)
        k_means.fit(train_data)
        np.save('poker_env/ToyPoker/data/toypoker_final_kmeans_centers.npy', k_means.cluster_centers_)
    kmeans_centers = np.load('poker_env/ToyPoker/data/toypoker_final_kmeans_centers.npy')
    ehs = table['ehs'].to_numpy()
    labels = np.argmin(np.absolute(np.tile(kmeans_centers, np.size(ehs)) - ehs), axis=0)
    new_label_column = pd.DataFrame({'label': labels})
    table.update(new_label_column)
    table.to_csv('poker_env/ToyPoker/data/toypoker_final_ehs.csv', sep=',', index=False)


def first_kmeans(n_clusters):

    # train_data = generate_train_data()
    train_data = np.load('poker_env/ToyPoker/data/toypoker_first_kmeans_train_data.npy', allow_pickle=True)
    # distance_matrix = calc_distance_matrix()
    distance_matrix = np.load('poker_env/ToyPoker/data/toypoker_final_cluster_distance_matrix.npy')
    k_means = KMeansWithEMD(200, 'custom', distance_matrix)
    k_means.fit(train_data)
    np.save('poker_env/ToyPoker/data/toypoker_first_kmeans_centers.npy', k_means.cluster_centers_)
    np.save('poker_env/ToyPoker/data/toypoker_first_kmeans_labels.npy', k_means.labels_)


def generate_train_data():

    table = pd.read_csv('poker_env/ToyPoker/data/toypoker_first_potential.csv', index_col=None, low_memory=False)
    n = len(table.index)
    train_data = []
    cur_state = table['cards_str'][0]
    cur_potential = []
    for i in range(n):
        if table['cards_str'][i] == cur_state:
            cur_potential.append(table['potential'][i])
        else:
            train_data.append(cur_potential)
            cur_state = table['cards_str'][i]
            cur_potential = [table['potential'][i]]
    np.save('poker_env/ToyPoker/data/toypoker_first_kmeans_train_data.npy', train_data)
    return np.asarray(train_data)


def calc_distance_matrix():

    means = np.reshape(get_cluster_mean(), (1, -1))
    distance_matrix = np.absolute(np.tile(np.transpose(means), np.size(means)) - means)
    np.save('poker_env/ToyPoker/data/toypoker_final_cluster_distance_matrix.npy', distance_matrix)
    return distance_matrix


def get_cluster_mean():

    table = pd.read_csv('poker_env/ToyPoker/data/toypoker_final_EHS.csv', index_col=None, low_memory=False)
    cluster_list = sorted(list(set(table['label'])))
    mean_list = []
    for cluster in cluster_list:
        cluster_ehs = table.loc[(table['label'] == cluster), 'ehs']
        cluster_mean = cluster_ehs.sum()/len(cluster_ehs)
        mean_list.append(cluster_mean)
    return mean_list
