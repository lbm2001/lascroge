import numpy as np
import os

def generate_tree_adj_matrix(num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.int32)
    for i in range(1, num_nodes):
        parent = np.random.randint(0, i)  # connect each new node to a previous one
        adj[i, parent] = 1
        adj[parent, i] = 1
    return adj

# Directory to save the .npy files
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)

# Parameters
num_graphs = 10        # Number of robot graphs
num_nodes = 5          # Number of joints per graph
vocab_size = 20        # Number of possible joint types (discrete IDs)

# === Feature Matrix (feat.npy) ===
# Each graph has `num_nodes` joints with discrete types (integers)
feat_data = np.random.randint(0, vocab_size, size=(num_graphs, num_nodes)).astype(np.int32)

# === Adjacency Matrix (adj.npy) ===
# Each graph has a symmetric 5x5 adjacency matrix indicating which joints are connected
adj_data = [generate_tree_adj_matrix(num_nodes) for _ in range(num_graphs)]
adj_data = np.stack(adj_data)

# === Save to .npy files ===
np.save(os.path.join(save_dir, "feat.npy"), feat_data)
np.save(os.path.join(save_dir, "adj.npy"), adj_data)

print("Dummy dataset saved to:", save_dir)