import numpy as np
import torch.nn as nn
import torch
from vae import VAE
from tree_batch_processor import TreeBatchProcessor

# CONSTANTS (attributes in the model)
FEATURE_DIM = 3
HIDDEN_SIZE = 300
LATENT_SIZE = 16
DEPTHT = 3
ENCODING_METHOD = "average"
MAX_NB = 4
FEATURE_DIM = 3

torch.manual_seed(42)

beta = 0.001 
alpha = 1.0
gamma = 1.0
num_epochs = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, MAX_NB, FEATURE_DIM, ENCODING_METHOD).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Graph 1: Linear chain (5 nodes) - 0-1-2-3-4
adj1 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 1],
      [0, 0, 0, 1, 0]
  ]
feats1 = [
      [1, 1, 1],
      [2, 2, 2],
      [3, 3, 3],
      [4, 4, 4],
      [5, 5, 5]
  ]

  # Graph 2: Star with 4 leaves (5 nodes) - 0 center, 1,2,3,4 leaves
adj2 = [
      [0, 1, 1, 1, 1],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0]
  ]
feats2 = [
      [0, 5, 0],  # Center node
      [1, 1, 4],  # Leaf 1
      [2, 2, 3],  # Leaf 2
      [3, 3, 2],  # Leaf 3
      [4, 4, 1]   # Leaf 4
  ]

  # Graph 3: Binary tree (5 nodes) - 0 root, 1,2 children of 0, 3,4 children of 1
adj3 = [
      [0, 1, 1, 0, 0],
      [1, 0, 0, 1, 1],
      [1, 0, 0, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0]
  ]
feats3 = [
      [2, 3, 2],  # Root
      [1, 2, 3],  # Left child of root
      [3, 4, 1],  # Right child of root
      [0, 1, 4],  # Left child of node 1
      [2, 1, 5]   # Right child of node 1
  ]

  # Graph 4: Path with branch (5 nodes) - 0-1-2-3 with 4 connected to 2
adj4 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 1],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0]
  ]
feats4 = [
      [1, 0, 2],
      [2, 1, 3],
      [3, 2, 4],  # Branch point
      [4, 3, 5],
      [3, 4, 3]   # Branch leaf
  ]

  # Graph 5: Y-shaped tree (5 nodes) - 0-1-2 spine, with 3,4 connected to 2
adj5 = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 1],
      [0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0]
  ]
feats5 = [
      [0, 2, 1],
      [1, 3, 2],
      [2, 4, 3],  # Junction point
      [3, 5, 2],  # Branch 1
      [1, 6, 4]   # Branch 2
  ]
"""
adj1 = [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0]
        ]

feats1 = [
    [1, 5, 6],
    [1, 3, 4],
    [0, 2, 4]
]

adj2 = [
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0]
]

feats2 = [
    [2, 7, 8],
    [0, 1, 2],
    [3, 5, 1]
]
"""

cur_conn = [np.array(adj1), np.array(adj2), np.array(adj3), np.array(adj4), np.array(adj5)]
cur_attr = [np.array(feats1), np.array(feats2), np.array(feats3), np.array(feats4), np.array(feats5)]

def tree_to_adjacency(tree_root):
    """
    Convert decoded TreeNode back to adjacency matrix by BFS.
    """
    # BFS to assign indices
    node_to_idx = {}
    nodes = []
    queue = [tree_root]
    idx = 0
    # BFS to index every node
    while queue:
        node = queue.pop(0)
        if node not in node_to_idx:
            node_to_idx[node] = idx
            nodes.append(node)
            idx += 1
            for neighbor in node.neighbors:
                if neighbor not in node_to_idx:
                    queue.append(neighbor)
    
    num_nodes = len(nodes) # Wenn Input Shape = Output Shape sein soll, dann hier num_nodes entfernen und als Parameter in die Methode übergeben
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for node, i in node_to_idx.items():
        for neighbor in node.neighbors:
            j = node_to_idx[neighbor]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    
    return adj_matrix

def train_loop(cur_conn, cur_attr):

    for epoch in range(num_epochs):
        tree_batch_processor = TreeBatchProcessor(cur_conn, cur_attr)
        tree_batch = tree_batch_processor.get_batch()
        encoding_holder = tree_batch_processor.prepare_encoding()

        model.zero_grad()
        loss, kl_div, wacc, tacc, pred_loss = model.forward(tree_batch, encoding_holder, beta, alpha, gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 50.0)
        optimizer.step()

        if(epoch % 50 == 0):
            print(f"Epoch {epoch}: Loss={loss.item(): .4f}, Pred Acc={wacc}, Stop Acc={tacc}, PredLoss={pred_loss.item(): .4f}, KL Divergence={kl_div.item(): .4f}")
            
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved after epoch", epoch)

def test_decoder( cur_attr, cur_conn):
    model = VAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, MAX_NB, FEATURE_DIM).to(device)
    model.load_state_dict(torch.load("trained_model.pth"))

    tb_processor = TreeBatchProcessor(cur_conn, cur_attr)
    encoding_holder = tb_processor.prepare_encoding()

    tree_vecs, messages = model.encoder.encode(encoding_holder)

    z_tree_vecs, _  = model.encoder.rsample(tree_vecs)
    #z_tree_vecs = model.encoder.mean_neural_network(tree_vecs)
    #z_single = z_tree_vecs[0:1]
    #root, all_nodes = model.decoder.decode(z_single, prob_decode=False, max_decode_len=100)
    for i in range(0,5):
        z_single = z_tree_vecs[i:i+1]
        root, all_nodes = model.decoder.decode(z_single, prob_decode=False, max_decode_len=100)
        print("Tree structure for batch item", i+1)
        print("Decoded tree structure:")
        print("Number of nodes in decoded tree:", len(all_nodes))
        for i, node in enumerate(all_nodes):
            print(f"Node {i}: {node.features}")
        adj_matrix = tree_to_adjacency(root)
        print("Adjacency Matrix of decoded tree:")
        print(adj_matrix)


train_loop(cur_conn, cur_attr)
#test_decoder("/Users/lukasmueller/github/lascroge/trained_model.pth", cur_attr, cur_conn)
test_decoder(cur_attr, cur_conn)