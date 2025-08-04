import numpy as np
import torch.nn as nn
import torch
from vae import VAE
from tree_batch_processor import TreeBatchProcessor

# CONSTANTS (attributes in the model)
FEATURE_DIM = 3
HIDDEN_SIZE = 100
LATENT_SIZE = 28
DEPTHT = 3
ENCODING_METHOD = "average"
MAX_NB = 4
FEATURE_DIM = 3


beta = 0.0
alpha = 1.0
gamma = 0.5
num_epochs = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, MAX_NB, FEATURE_DIM, ENCODING_METHOD).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

cur_conn = [np.array(adj1), np.array(adj2)]
cur_attr = [np.array(feats1), np.array(feats2)]

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

def test_decoder(model_path, cur_attr, cur_conn):
    model = VAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, MAX_NB, FEATURE_DIM).to(device)
    model.load_state_dict(torch.load(model_path))

    tb_processor = TreeBatchProcessor(cur_conn, cur_attr)
    encoding_holder = tb_processor.prepare_encoding()

    res = model.encoder.encode(encoding_holder)
    tree_vecs = res[0]

    z_tree_vecs, _  = model.encoder.rsample(tree_vecs)
    root, all_nodes = model.decoder.decode(z_tree_vecs, prob_decode=False, max_decode_len=100)
    print(root)
    print(all_nodes)


train_loop(cur_conn, cur_attr)
#test_decoder("/Users/lukasmueller/github/lascroge/trained_model.pth", cur_attr, cur_conn)
    





