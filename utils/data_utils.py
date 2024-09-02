import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import itertools as it
from .general_utils import *

def str2int(x):
    # [list(map(int, x)) for x in X] is an alternative
    return [[int(column) for column in row] for row in x]

def get_pdist(mean=0, nbins=1):
    pdf = np.histogram(rng.normal(loc=mean, size=1000), bins=nbins)[0]
    return pdf/np.sum(pdf)

def gen_binary(data, N, mu=None):
    if mu is None: # uniform random
        bd = rng.choice(data, size=N)
    else: # weighted normal distribution
        pdf = get_pdist(mean=mu, nbins=len(data))
        bd = rng.choice(data, size=N, p=pdf)
    return bd

def gen_ohe_columns():
    # create some lists of values for 'binary numbers' simulating one-hot encodings of common categorical data
    # TODO make these arrays/lists
    d1 = [[0], [1]]  # simple binary choice - True or False

    # at least one and mutually exclusive values
    d2r = str2int([f'{i:02b}' for i in [1, 2]])  # logically equivalent to d1
    d3r = str2int([f'{i:03b}' for i in [1, 2, 4]])
    d4r = str2int([f'{i:04b}' for i in [1, 2, 4, 8]])
    d5r = str2int([f'{i:05b}' for i in [1, 2, 4, 8, 16]])
    d6r = str2int([f'{i:06b}' for i in [1, 2, 4, 8, 16, 32]])
    d7r = str2int([f'{i:07b}' for i in [1, 2, 4, 8, 16, 32, 64]])
    d8r = str2int([f'{i:08b}' for i in [1, 2, 4, 8, 16, 32, 64, 128]])

    # mutually exclusive values but including no value
    d2r0 = str2int([f'{i:02b}' for i in [0, 1, 2]])
    d3r0 = str2int([f'{i:03b}' for i in [0, 1, 2, 4]])
    d4r0 = str2int([f'{i:04b}' for i in [0, 1, 2, 4, 8]])
    d5r0 = str2int([f'{i:05b}' for i in [0, 1, 2, 4, 8, 16]])
    d6r0 = str2int([f'{i:06b}' for i in [0, 1, 2, 4, 8, 16, 32]])
    d7r0 = str2int([f'{i:07b}' for i in [0, 1, 2, 4, 8, 16, 32, 64]])
    d8r0 = str2int([f'{i:08b}' for i in [0, 1, 2, 4, 8, 16, 32, 64, 128]])

    # any combination of n values including 0 = none of the above
    d2 = [list(i) for i in it.product([0, 1], repeat=2)]
    d3 = [list(i) for i in it.product([0, 1], repeat=3)]
    d4 = [list(i) for i in it.product([0, 1], repeat=4)]
    d5 = [list(i) for i in it.product([0, 1], repeat=5)]
    d6 = [list(i) for i in it.product([0, 1], repeat=6)]
    d7 = [list(i) for i in it.product([0, 1], repeat=7)]
    d8 = [list(i) for i in it.product([0, 1], repeat=8)]
    return (d1, d2, d3, d4, d5, d6, d7, d8, d2r, d3r, d4r, d5r, d6r, d7r, d8r, d2r0, d3r0, d4r0, d5r0, d6r0, d7r0, d8r0)

def parse_config(config):
    # extract the relevant columns
    cluster_sizes = [col[0] for col in config]
    centres = [col[1] for col in config]
    # variance list defines the number of clusters
    # all covariance matrices are multiples of the identity
    cluster_variance = [col[2] for col in config]
    num_clusters = len(cluster_variance)
    return cluster_sizes, centres, cluster_variance, num_clusters

def generate_mixed_data(config,
                        num_continuous_features,
                        categorical_features,
                        use_make_blobs=True):
    rng = np.random.default_rng(seed=1234)

    cluster_sizes, centres, cluster_variance, num_clusters = parse_config(config)

    # Initialize data storage
    continuous_data = []
    # labels = []
    df_bin = pd.DataFrame()
    df_labels = pd.DataFrame()

    if use_make_blobs:
        # print(f"{points_per_cluster} {num_clusters} {variance} {num_continuous_features}")
        # print(points_per_cluster*num_clusters)
        continuous_data, cluster_idx = make_blobs(
            n_samples=cluster_sizes
            , centers=centres
            , cluster_std=cluster_variance
            , n_features=num_continuous_features
            , shuffle=False
            , random_state=seed
        )
    else:
        covariances = []

    # Generate binary data for each cluster
    for i in range(num_clusters):
        if not use_make_blobs:
            cov = np.eye(num_clusters) * cluster_variance[i]
            # Continuous data
            cluster_continuous_data = rng.multivariate_normal(mean=centres[i], cov=cov, size=cluster_sizes[i])
            continuous_data.append(cluster_continuous_data)

        # Categorical data
        # based on list of binary values
        feature_data = pd.DataFrame()
        for binary_data in categorical_features:  # build out columns
            gb = pd.DataFrame(gen_binary(binary_data, cluster_sizes[i], mu=0.2 * i))
            # gb = pd.DataFrame(gen_binary(binary_data, points_per_cluster))
            feature_data = pd.concat([feature_data, gb], axis=1, ignore_index=True)

        # append rows to binary data
        df_bin = pd.concat([df_bin, feature_data], axis=0).reset_index(drop=True)

        # Labels
        if use_make_blobs:
            cluster_labels = pd.DataFrame(cluster_idx)
        else:
            labels = [i] * cluster_sizes[i]
            cluster_labels = pd.DataFrame(labels)

        df_labels = pd.concat([df_labels, cluster_labels], axis=0).reset_index(drop=True)

    # Combine continuous and categorical data
    if use_make_blobs:
        df_cont = pd.DataFrame(continuous_data)
    else:
        df_cont = pd.DataFrame(continuous_data[0])
    df_mixed = pd.concat([df_cont, df_bin], axis=1)

    num_binary_features = sum([len(categorical_features[i][0]) for i in range(len(categorical_features))])
    # List of column names for continuous and categorical
    continuous_columns = [f'Continuous_Feature_{i + 1}' for i in range(num_continuous_features)]
    categorical_columns = [f'Binary_Feature_{i + 1}' for i in range(num_binary_features)]

    # Update column names
    df_mixed.columns = [*continuous_columns, *categorical_columns]

    # add cluster true value to dataframe
    df_mixed['ClusterTruth'] = df_labels

    # standardise the continuous data using sklearn
    scale = MinMaxScaler()
    df_mixed[continuous_columns] = scale.fit_transform(df_mixed[continuous_columns])

    return (df_mixed, continuous_columns, categorical_columns)

d1, d2, d3, d4, d5, d6, d7, d8, d2r, d3r, d4r, d5r, d6r, d7r, d8r, d2r0, d3r0, d4r0, d5r0, d6r0, d7r0, d8r0 = gen_ohe_columns()

def count_ohe(categorical_features):
    num_binary_features = sum([len(categorical_features[i][0]) for i in range(len(categorical_features))])
    return num_binary_features
