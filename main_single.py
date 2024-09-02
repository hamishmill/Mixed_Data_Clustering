from pathlib import Path

from gbppm import run_ppm
from generate_data import generate_data
from generate_plots import generate_plots
from utils.data_utils import *

do_generate_data = False
do_run_ppm = False
do_generate_plots = True

# Initialise synthetic data configuration

num_continuous_features = 2
categorical_features = [] # list of binary options as generated above
cluster_config = [
        [333, [0,1], 1.2],
        [333, [4,4], 1.2],
        [334, [0,6], 1.2]
    ]
num_clusters = len(cluster_config)

if do_generate_data:
    filename = generate_data(cluster_config, num_continuous_features, categorical_features)

if do_run_ppm:
    # if running one example start to finish
    # csv_filename = Path(filename).stem

    # or manually enter an alternative
    csv_filename = 'N1000_V1.2_K3_Cns2_Cat0' # manually enter an alternative

    num_iterations = 2000
    num_cat = count_ohe(categorical_features)
    data_split = [num_continuous_features, num_cat, 0]
    output_list = run_ppm(csv_filename, data_split, num_iterations, num_clusters)

if do_generate_plots:
    generate_plots()
