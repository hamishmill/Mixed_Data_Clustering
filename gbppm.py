import os
import pandas as pd
from time import strftime
from sklearn.metrics import pairwise_distances

from utils.gbppm_utils import *
from utils.general_utils import *
from utils.plot_utils import likelihood_to_rgb

def run_ppm(csv_filename, duo, num_iterations, num_clusters):

    # run_sampler(csv_filename, duo, num_iterations, num_clusters, lambda_supplied = None, weights = [1,0,0]) # single run using default weights and lambda

    for lambda_given in [10,100]:
        run_sampler(csv_filename, duo, num_iterations, num_clusters, lambda_given)

    # weights_list = [
    #     None
    #     , [0.25, 0.75, 0]
    #     # , [0.5, 0.5, 0]
    #     , [0.75, 0.5, 0]
    # ]

    weights_list = [
        None
        , [3, 1, 0]
        , [1, 3, 0]
    ]

    # run lambda sampled from gamma for each weight
    # for weights in weights_list:
    #     run_sampler(csv_filename, duo, num_iterations, num_clusters, None, weights)

    # run fixed lambda for each weight
    # for lambda_given in [10, 50, 100]:
    #     for weights in weights_list:
    #         run_sampler(csv_filename, duo, num_iterations, num_clusters, lambda_given, weights)

    # run fixed lambda for each weight
    # for lambda_given in [10, 50, 100]:
    #     run_sampler(csv_filename, duo, num_iterations, num_clusters, lambda_supplied=lambda_given, weights=[0,1,0])

def run_sampler(csv_filename, duo, num_iterations, num_clusters, lambda_supplied=None, weights=None):

    subfolder = 'data'
    # filename = f'N{N}_K{num_clusters}_Cns{num_continuous}_Cat{num_binary_features}.csv'
    filename = os.path.join(subfolder, csv_filename)
    df = pd.read_csv(f"{filename}.csv")

    K = num_clusters  # number of clusters
    N = len(df)

    if weights is None:
        weights_distance = [1, 1, 0]
        w_name = 'eq'
    else:
        weights_distance = weights
        w_name = weights_distance[0]

    # use the known structure of the data to extract continuous, unordered and ordered data
    # dimensions defined by list duo [d,u,o]
    # d continuous columns, u unordered columns, o ordered columns
    # duo = [2,1,0]
    (D,d,u,o,continuous,unordered,ordered) = parse_data_structure(duo)

    # Convert categorical to Boolean - avoids warning on other metrics
    # if unordered is not None:
    #     df[df.columns[unordered]] = df[df.columns[unordered]].astype('boolean')

    # convert from dataframe to numpy arrays
    data_d = np.array(df.iloc[:, continuous]) if continuous is not None else None
    data_u = np.array(df.iloc[:, unordered]) if unordered is not None else None
    data_o = np.array(df.iloc[:, ordered]) if ordered is not None else None

    # can't think of any sensible way to express the dissimilarity from (x,y) to say, colour
    # so will treat each class of data as a block with a suitable metric for each data class
    pwd_d, pwd_u, pwd_o = None, None, None

    pwd_d = pairwise_distances(data_d, metric='sqeuclidean', n_jobs=-1) if data_d is not None else None
    pwd_u = pairwise_distances(data_u, metric='hamming', n_jobs=-1) if data_u is not None else None
    pwd_o = pairwise_distances(data_o, metric='manhattan', n_jobs=-1) if data_o is not None else None

    ###
    # Combine Distances
    ###
    pwd = combine_distances(pwd_d, pwd_u, pwd_o, weights_distance)

    # randomly allocate cluster
    cluster_map = (rng.integers(0, high=K, size=N))

    # initial calculation of loss is a one off
    # subsequently it is just an update after either adding or removing a row
    # need to store cluster map k, loss for each cluster, number of entries in cluster nk
    cluster_loss = np.zeros(K, dtype=float)
    cluster_count = np.zeros(K, dtype=int)
    for k in range(K):  # for each cluster
        mask = (cluster_map == k)
        mask2d = np.outer(mask, mask)
        cluster_count[k] = np.sum(mask)
        cluster_loss[k] = np.sum(pwd[mask2d]) / (2 * cluster_count[k])

    total_loss = np.sum(cluster_loss)

    #  Initialisations
    burnin = num_iterations // 10
    alpha = 0
    beta = 0

    if lambda_supplied is None:
        lambda_p = rng.gamma(alpha + N * D, 1 / (beta + total_loss))
    else:
        lambda_p = lambda_supplied

    # arrays to hold history of each iteration
    lambda_history = np.zeros(num_iterations, dtype=float)
    loss_history = np.zeros(num_iterations, dtype=float)
    cluster_history = np.zeros((num_iterations, N), dtype=float)
    # probability_history will be rather large
    probability_history = np.zeros((num_iterations, N, K), dtype=float)
    log_prob_history = np.zeros((num_iterations, N, K), dtype=float)

    #
    # Main sampling loop
    #
    print(f"{strftime('%H:%M:%S')} processing {csv_filename}")
    for j in range(num_iterations):
        if j % (num_iterations // 8) == 0:  # report update about 8 times
            print(f"{strftime('%H:%M:%S')} iteration {j + 1}")
        probability_i = np.zeros((N, K))
        log_prob_i = np.zeros((N, K))

        for i in range(N):
            prev_k = cluster_map[i]
            loss_with = np.zeros(K, dtype=float)
            loss_without = np.zeros(K, dtype=float)
            delta = np.zeros(K, dtype=float)
            log_prob = np.zeros(K, dtype=float)
            # loop round all clusters adding point i to each
            for k in range(K):
                mask = (cluster_map == k)
                n = cluster_count[k]

                if prev_k == k:  # in cluster
                    loss_with[k] = cluster_loss[k]
                    mean_row_i = np.mean(pwd[i][mask])
                    loss_without[k] = n / (n - 1) * (cluster_loss[k] - mean_row_i)
                else:  # not in cluster
                    loss_without[k] = cluster_loss[k]
                    mask[i] = True  # add point i to mask for cluster k temporarily
                    mean_row_i = np.mean(pwd[i][mask])  # calculate the mean of row i for cluster k
                    loss_with[k] = n / (n + 1) * cluster_loss[k] + mean_row_i

                delta[k] = loss_with[k] - loss_without[k]

            log_prob = -lambda_p * delta  # can exp vectorised outside loop

            log_prob = log_prob - max(log_prob)  # stop it getting too big for exp?
            probability = np.exp(log_prob)  # vectorised exp
            if np.sum(probability) == 0:
                raise ValueError('Probabilities sum to 0')
            else:
                probability = probability / np.sum(probability)

            # collect intermediate arrays
            probability_i[i] = probability
            log_prob_i[i] = log_prob

            # simply allocate
            # new_k = np.argmin(log_prob)

            # or Gibbs sample
            new_k = rng.choice(range(0, K), p=probability)

            # update tracking arrays
            if new_k != prev_k:  # move point i to cluster new_k
                cluster_count[prev_k] = cluster_count[prev_k] - 1
                cluster_loss[prev_k] = loss_without[prev_k]
                # persist change to cluster
                cluster_map[i] = new_k
                cluster_count[new_k] = cluster_count[new_k] + 1
                cluster_loss[new_k] = loss_with[new_k]

        total_loss = np.sum(cluster_loss)
        # update lambda sampled from Gamma distribution
        # but only if total_loss is greater than 0 (possible invalid scale otherwise)
        if lambda_supplied is None and total_loss > 0:
            lambda_p = rng.gamma(alpha + N * D, 1 / (beta + total_loss))

        # save the values for each iteration
        lambda_history[j] = lambda_p
        loss_history[j] = total_loss
        cluster_history[j] = np.array(cluster_map)
        probability_history[j] = np.array(probability_i)
        log_prob_history[j] = np.array(log_prob_i)

    print(f"{strftime('%H:%M:%S')} iteration {j + 1}")
    print(f"{strftime('%H:%M:%S')} Done")

    #
    # Post-processing
    #
    ### Remove burn in from history files
    probability_history = probability_history[burnin:]

    # calculate average probability
    avg_probability = np.mean(probability_history, axis=0)

    #
    # Save output
    #
    # Make a copy of the data so other columns can be added later
    data_plot = np.copy(data_d)
    data_plot = np.column_stack((data_plot, data_u)) if data_u is not None else data_plot
    data_plot = np.column_stack((data_plot, data_o)) if data_o is not None else data_plot
    # add the final cluster values
    data_plot = np.column_stack((data_plot, cluster_map))
    # Move the data to a dataframe to simplify manipulation
    # e.g. can add column titles back in
    dfp = pd.DataFrame(data_plot, columns=[*df.columns[:D], "Cluster"])  # first D columns are the date
    dfp = dfp.astype({"Cluster": int})

    dfp["ClusterTruth"] = df["ClusterTruth"]

    colors_rgb = likelihood_to_rgb(avg_probability)
    dfp['Colour'] = list(colors_rgb)

    cluster_truths = np.array(pd.get_dummies(df['ClusterTruth']).astype(int))
    cluster_probabilities = probability_history[-1]
    RI = calc_RI(cluster_truths, cluster_probabilities, num_clusters=3)

    #
    # Save to output
    #
    if lambda_supplied is None:
        lambda_p = 'gamma'
    base_filename = f'{csv_filename}_Itr{num_iterations}_L{lambda_p}_W{w_name}'
    append_to_file("Results.txt", f"{base_filename}: RI:{RI} lambda final: {lambda_history[-1]}")

    csv_filename = os.path.join('output', f"{base_filename}.csv")
    dfp.to_csv(csv_filename, index=False)

    # save other arrays
    npz_filename = os.path.join('output_npz', base_filename) # defaults to npz extension
    np.savez( npz_filename
             , cluster_map
             ,lambda_history
             , loss_history
             , cluster_history
             , probability_history
             , log_prob_history)
