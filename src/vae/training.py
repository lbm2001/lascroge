import numpy as np
import torch.nn as nn
import torch
from vae import VAE
from helper import tensorize

# CONSTANTS (attributes in the model)
HIDDEN_SIZE = 3
LATENT_SIZE = 28
DEPTHT = 3
ENCODING_METHOD = "average"
MAX_NB = 4


beta = 0.0
alpha = 1.0
gamma = 0.5
num_epochs = 5000


model = VAE(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, MAX_NB, ENCODING_METHOD)
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

for epoch in range(num_epochs):

    batch = tensorize(cur_attr, cur_conn)
    tree_batch, encoding_holder = batch
    model.zero_grad()
    loss, kl_div, wacc, pred_loss = model.forward(tree_batch, encoding_holder, beta, alpha, gamma)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 50.0)
    optimizer.step()

    if(epoch % 50 == 0):
            print(f"Epoch {epoch}: Loss={loss.item(): .4f}, Pred Acc={wacc}, Stop Acc={tacc}, PredLoss={pred_loss.item(): .4f}, KL Divergence={kl_div.item(): .4f}")
        
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved after epoch", epoch)






