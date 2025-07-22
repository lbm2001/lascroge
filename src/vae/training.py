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

model.zero_grad()
tree_batch, encoding_holder = tensorize(cur_attr, cur_conn)
loss, kl_div, wacc, pred_loss = model.forward(tree_batch, encoding_holder, beta, alpha, gamma)
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), 50.0)
optimizer.step()