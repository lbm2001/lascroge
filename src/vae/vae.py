import torch.nn as nn
import torch
from encoder import Encoder
from decoder import Decoder

HIDDEN_SIZE = 8
LATENT_SIZE = 28
DEPTHT = 3
ENCODING_METHOD = "average"


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(HIDDEN_SIZE, LATENT_SIZE, DEPTHT, ENCODING_METHOD)
        self.decoder = Decoder(HIDDEN_SIZE, LATENT_SIZE)

        self.mean_neural_network = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)
        self.var_neural_network = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)
    

    def forward(self, tree_data, encoding_holder, beta, alpha, gamma):
        tree_vectors, tree_messages = self.encoder.encode(encoding_holder)
        latent_space_tree_vecs, kl_divergence = self.encoder.rsample(tree_vectors, self.mean_neural_network, self.var_neural_network)

        pred_loss = torch.tensor(0.0)
        
        word_loss, topo_loss, word_acc, topo_acc = self.decoder.forward(tree_data, latent_space_tree_vecs)
        
        return alpha * word_loss + topo_loss + beta * kl_divergence + gamma * pred_loss, kl_divergence.item(), \
                word_acc, topo_acc, pred_loss.item()

    