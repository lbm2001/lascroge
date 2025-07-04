import os
import torch
import numpy as np
from fast_jtnn import JTNNVAE, tensorize

# --- Configuration ---
MODEL_PATH = "results/model.iter-1000"  # Replace with your checkpoint path
DATA_DIR = "glso\data\robot_graphs"               # Replace with your data directory
LATENT_SIZE = 28                             # Use the same as in training
HIDDEN_SIZE = 450
DEPTHT = 20
ENCODE = "sum"
PRED = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = JTNNVAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, ENCODE, PRED).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Load a sample input ---
#adj = np.load(os.path.join(DATA_DIR, "adj.npy"), allow_pickle=True)
#feat = np.load(os.path.join(DATA_DIR, "feat.npy"), allow_pickle=True)
adj = np.load(os.path.join(DATA_DIR, "simple_robot.npy"), allow_pickle=True)
feat = np.load(os.path.join(DATA_DIR, "simple_robot_features.npy"), allow_pickle=True)

# Use the first sample for testing
print("Adjacency matrix shape:", adj.shape)
print("Feature matrix shape:", feat.shape)
"""
sample_attr = feat[0:1]
sample_conn = adj[0:1]#
print("Sample attribute shape:", sample_attr)
print("Sample connection shape:", sample_conn)

batch = tensorize(sample_attr, sample_conn)

# --- Encode and Decode (Reconstruction) ---
with torch.no_grad():
    # Encode to latent space
    #print("Input batch shape:", batch[1])
    tree_vecs, tree_mess = model.encode(batch[1])

    # Sample latent vector
    z_tree_vecs, kl_loss = model.rsample(tree_vecs, model.T_mean, model.T_var)
    # Decode from latent space
    recon = model.decode(z_tree_vecs, prob_decode=True)
    print("\n--- Input and Output Comparison ---")
    print("Original input (first feature row):", sample_attr[0])
    print("Reconstructed output:", recon[0])

# --- Sample from latent space ---
with torch.no_grad():
    z_sample = torch.randn(1, LATENT_SIZE).to(device)
    sampled = model.decode(z_sample, prob_decode=True)
    print("Sampled output from latent space:", sampled)
    print("Latent vector (from input):", z_tree_vecs)
    print("Random latent vector (sampled):", z_sample)
    """