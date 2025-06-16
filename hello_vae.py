import torch
import torch.nn as nn
import torch.optim as optim

from fast_jtnn.jtnn_vae import JTNNVAE  # Adjust path as needed
from fast_jtnn.params import VOCAB_SIZE

class DummyNode:
    def __init__(self):
        self.idx = 0
        self.neighbors = []
        self.wid = 1  # word ID / label

class DummyTree:
    def __init__(self):
        self.nodes = [DummyNode()]

# Dummy config
hidden_size = 32
latent_size = 16
depthT = 3
encoding = 'average'
pred_prop = False  # Set to True to test contact prediction

# Instantiate model
model = JTNNVAE(hidden_size, latent_size, depthT, encoding, pred_prop)
device = torch.device("cpu")
model.to(device)
model.train()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Dummy Data Setup ===
# Shapes
BATCH_SIZE = 2
NUM_NODES = 4
NUM_MESSAGES = 6
hidden_size = 32

# Dummy input features
fnode = torch.randint(0, VOCAB_SIZE, (NUM_NODES * BATCH_SIZE,))  # token IDs
fmess = torch.randint(0, NUM_NODES * BATCH_SIZE, (NUM_MESSAGES,))  # indexes into fnode

# Graph connections (messages to nodes and messages to messages)
node_graph = torch.randint(0, NUM_MESSAGES, (NUM_NODES * BATCH_SIZE, 3))  # each node gets 3 messages
mess_graph = torch.randint(0, NUM_MESSAGES, (NUM_MESSAGES, 3))  # each message gets 3 neighbors

# Tree structure indexing
scope = [(0, NUM_NODES), (NUM_NODES, NUM_NODES)]  # 2 trees of 4 nodes
leafs = [[1, 2], [5, 6]]  # dummy leaf node indices per tree

# Device push
fnode = fnode.to(device)
fmess = fmess.to(device)
node_graph = node_graph.to(device)
mess_graph = mess_graph.to(device)

# Bundle full jtenc_holder
jtenc_holder = (fnode, fmess, node_graph, mess_graph, scope, leafs)

x_data = [DummyTree(), DummyTree()]  # Just placeholders
loss, kl, word_acc, topo_acc, _ = model((x_data, jtenc_holder), None, beta=0.1, alpha=1.0, gamma=0.0)

print(f"Loss: {loss.item():.4f} | KL: {kl:.4f} | Word Acc: {word_acc:.4f} | Topo Acc: {topo_acc:.4f}")

# === Backward + optimize ===
optimizer.zero_grad()
loss.backward()
optimizer.step()