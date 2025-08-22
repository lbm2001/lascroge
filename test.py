import numpy as np

adj = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\adj.npy", allow_pickle=True)
feat = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\feat.npy", allow_pickle=True)
print(adj[0].shape)
print(feat[0].shape)
# Make both matrices the same size (use smaller dimension)
for i in range(len(adj)):
    min_size = min(adj[i].shape[0], feat[i].shape[0])
    adj[i] = adj[i][:min_size, :min_size]
    feat[i] = feat[i][:min_size, :]

print(adj[0].shape)
print(feat[0].shape)
a0 = adj[0]
print(len(adj))