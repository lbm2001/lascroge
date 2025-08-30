import numpy as np
import os

np.random.seed(42)

def generate_tree_with_features(num_nodes, feature_dim):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for i in range(1, num_nodes):
        parent = np.random.randint(0, i)  # connect each new node to a previous one
        adj[i, parent] = 1
        adj[parent, i] = 1
    
    features = np.random.rand(num_nodes, feature_dim).astype(np.float32)
    mid = num_nodes // 2
    features[:mid, 0] = 1  # first half type 1
    features[mid:, 0] = 0  # second half type 0

    return adj, features

# === Parameters ===
num_graphs = 20        # number of graphs
min_nodes = 15       # minimum nodes per graph
max_nodes = 15         # maximum nodes per graph
feature_dim = 5        # dimension of each node feature vector
save_dir = "/Users/lukasmueller/github/lascroge/data/dummy_data"
os.makedirs(save_dir, exist_ok=True)

# === Generate dataset ===
adj_data = []
feat_data = []

for _ in range(num_graphs):
    num_nodes = np.random.randint(min_nodes, max_nodes + 1)
    adj, feat = generate_tree_with_features(num_nodes, feature_dim)
    adj_data.append(adj)
    feat_data.append(feat)

# Save as object arrays (since shapes vary)
np.save(os.path.join(save_dir, "adj.npy"), np.array(adj_data, dtype=np.float32))
np.save(os.path.join(save_dir, "feat.npy"), np.array(feat_data, dtype=np.float32))

print("Dummy dataset saved to:", save_dir)
print("Example adj shape:", adj_data[0].shape)
print("Example feat shape:", feat_data[0].shape)
