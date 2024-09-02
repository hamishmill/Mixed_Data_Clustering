from utils.data_utils import *
import os

def generate_data(config, num_continuous, categorical):

    # extract the relevant columns for naming purposes
    cluster_variance = [col[2] for col in config]
    num_clusters = len(cluster_variance)

    (df, continuous_columns, categorical_columns) = generate_mixed_data( config
                                                                        , num_continuous
                                                                        , categorical
                                                                        , use_make_blobs=True)
    # Save the dataframe
    N = len(df)
    num_binary_features = sum([len(categorical[i][0]) for i in range(len(categorical))])

    # construct filename
    subfolder = 'data'
    filename = f'N{N}_V{cluster_variance[0]}_K{num_clusters}_Cns{num_continuous}_Cat{num_binary_features}.csv'
    filename = os.path.join(subfolder, filename)

    # Write to file
    df.to_csv(filename, index=False)
    return filename