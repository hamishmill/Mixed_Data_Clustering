import numpy as np

seed = 1234
rng = np.random.default_rng(seed=seed)

def head(data, n=5):
    print(f"length: {len(data)}")
    return data[:n,:]

def append_to_file(file_path, text):
    try:
        with open(file_path, 'a') as file:
            file.write(text+ '\n')
    except Exception as e:
        print(f"Error: {e}")