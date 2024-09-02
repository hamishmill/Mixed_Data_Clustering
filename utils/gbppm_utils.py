import numpy as np

def parse_data_structure(duo):
    assert len(duo) == 3, "parse_structure: parameter must be a list of length 3"
    D = np.sum(duo)
    d=duo[0]
    u=duo[1]
    o=duo[2]
    continuous = range(0,d) if d > 0 else None
    unordered = range(d,d+u) if d != d+u else None
    ordered = range(d+u,D) if d+u != D else None
    return (D,d,u,o,continuous,unordered,ordered)

def combine_distances(pwd_d, pwd_u, pwd_o, weights = [1,1,1]):
    # simple linear combination with weights
    avg_weight = np.sum(weights)/3
    sum = 0
    sum = weights[0]*pwd_d if pwd_d is not None else sum
    sum = sum + weights[1]*pwd_u if pwd_u is not None else sum
    sum = sum + weights[2]*pwd_o if pwd_o is not None else sum
    return sum/avg_weight

def loss(pwd, mask):
    mask2d = np.outer(mask, mask)
    cluster_count = np.sum(mask)
    cluster_loss = np.sum(pwd[mask2d])/(2*cluster_count)
    return cluster_loss

##################################################
# fuzzy RAND
##################################################
def calculate_V_or_Y(data, num_classes):
    n = len(data)
    V = np.zeros((n, n))  # Initialize the V matrix

    # List of column names
    # column_names = data.columns
    # print(column_names[0])

    # Calc matrix
    for j1 in range(n):
        # print(f'j1: {j1}')
        for j2 in range(j1 + 1, n):  # so j2 is always higher than j1
            # print(f'j2: {j2}')
            max_values = []  # for s norm
            for i in range(num_classes):
                # For the current pair of points returns the min value so only 1 if both in the same class
                # t_norm = min(data.at[j1, column_names[i]], data.at[j2, column_names[i]])
                t_norm = min(data[j1, i], data[j2, i])
                max_values.append(t_norm)
            # print(max_values)
            # Calculate s-norm (max) across all t-norm results for this pair
            V[j1, j2] = max(max_values)
            V[j2, j1] = V[j1, j2]  # for symmetry

    return V


def calculate_X_or_Z(data, num_classes):
    n = len(data)
    X = np.zeros((n, n))  # Initialize the X matrix

    # column_names = df_encoded.columns

    # calc matrix
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            max_t_norms = []
            # Iterate through all combinations of different classes
            for i1 in range(num_classes):
                for i2 in range(num_classes):
                    if i1 != i2:
                        # Calculate t-norm (min) for the current class pair
                        t_norm = min(data[j1, i1], data[j2, i2])
                        # print(f'for point:{j1},{j2} at class pair:{i1},{i2} t norm = {t_norm}')
                        max_t_norms.append(t_norm)
            # print(max_t_norms)
            # Calculate s-norm (max) across all t-norm results for this pair
            X[j1, j2] = max(max_t_norms)
            X[j2, j1] = X[j1, j2]  # Since the matrix is symmetric

    return X


def compute_aggregates(V, X, Y, Z):
    V_Y = np.sum(V * Y)
    V_Z = np.sum(V * Z)
    X_Y = np.sum(X * Y)
    X_Z = np.sum(X * Z)

    return V_Y, V_Z, X_Y, X_Z


def calc_RI(cluster_truths, cluster_probabilities, num_clusters):
    V_matrix = calculate_V_or_Y(cluster_truths, num_clusters)
    X_matrix = calculate_X_or_Z(cluster_truths, num_clusters)
    Y_matrix = calculate_V_or_Y(cluster_probabilities, num_clusters)
    Z_matrix = calculate_X_or_Z(cluster_probabilities, num_clusters)

    matrices = [V_matrix, X_matrix, Y_matrix, Z_matrix]

    for i, matrix in enumerate(matrices):
        matrix = np.triu(matrix)
        matrix += matrix.T - np.diag(matrix.diagonal())
        matrices[i] = matrix  # Update the matrix in the list defined above

    V_matrix, X_matrix, Y_matrix, Z_matrix = matrices

    V_Y, V_Z, X_Y, X_Z = compute_aggregates(V_matrix, X_matrix, Y_matrix, Z_matrix)

    a = V_Y
    b = V_Z
    c = X_Y
    d = X_Z

    # RAND Value
    omega = (a + d) / (a + b + c + d)
    return omega