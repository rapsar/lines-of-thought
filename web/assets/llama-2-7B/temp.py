import numpy as np

trajectories = np.load("trajectories.npy")
singular_vectors = np.load("singular_vectors.npy")

print("Trajectories shape:", trajectories.shape)
print("Singular vectors shape:", singular_vectors.shape)