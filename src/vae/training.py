import numpy as np
import torch.nn as nn
import torch
from vae import VAE
from helper import tensorize

beta = 0.0
alpha = 1.0
gamma = 0.5

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

adj1 = [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
]

feats1 = [
    [1, 5, 6, 2, 3],
    [1, 3, 4, 5, 6],
    [0, 2, 4, 5, 6], 
    [2, 4, 7, 2, 1],
    [3, 6, 3, 1, 5]
]

adj2 = [
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
]

feats2 = [
    [1, 5, 6, 2, 3],
    [1, 3, 4, 5, 6],
    [0, 2, 4, 5, 6], 
    [2, 4, 7, 2, 1],
    [3, 6, 3, 1, 5]
]
cur_conn = [np.array(adj1), np.array(adj2)]
cur_attr = [np.array(feats1), np.array(feats2)]

model.zero_grad()
tree_batch, encoding_holder = tensorize(cur_attr, cur_conn)
loss, kl_div, wacc, pred_loss = model.forward(tree_batch, encoding_holder, beta, alpha, gamma)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 50.0)
optimizer.step()