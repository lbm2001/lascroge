import numpy as np

# Load the files
adj = np.load("./data/adj.npy")
feat = np.load("./data/feat.npy")

# Print the contents
print("adj shape:", adj.shape)
print(adj)

print("feat shape:", feat.shape)
print(feat)