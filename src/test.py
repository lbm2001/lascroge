import numpy as np

# Load adjacency matrices
adj_g1 = np.load('data/adj_g1.npy', allow_pickle=True).squeeze(0)
adj_g1 = adj_g1[1: , 1:]
adj_go1 = np.load('data/adj_go1.npy', allow_pickle=True).squeeze(0)
adj_go2 = np.load('data/adj_go2.npy', allow_pickle=True).squeeze(0)
adj_h1 = np.load('data/adj_h1.npy', allow_pickle=True).squeeze(0)

# Load feature matrices
feat_g1 = np.load('data/feat_g1.npy', allow_pickle=True)
feat_go1 = np.load('data/feat_go1.npy', allow_pickle=True)
feat_go2 = np.load('data/feat_go2.npy', allow_pickle=True)
feat_h1 = np.load('data/feat_h1.npy', allow_pickle=True)

print("Adjacency Matrices:")
print("adj_g1 shape:", adj_g1.shape)
print("adj_go1 shape:", adj_go1.shape)
print("adj_go2 shape:", adj_go2.shape)
print("adj_h1 shape:", adj_h1.shape)
print("\nFeature Matrices:")
print("feat_g1 shape:", feat_g1.shape)
print("feat_go1 shape:", feat_go1.shape)
print("feat_go2 shape:", feat_go2.shape)
print("feat_h1 shape:", feat_h1.shape)
# Example of how to use the loaded data
# Here you can add your processing logic or model training code

#print(adj_go1[0,:])
print("----------------------")
print(feat_g1[0 ,1, :])
print(feat_go1[0 ,1, :])
print(feat_h1[0 ,1, :])
print(feat_h1[0 ,1, :])