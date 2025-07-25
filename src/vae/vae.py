import torch.nn as nn
import torch
from encoder import Encoder
from decoder import Decoder


class VAE(nn.Module):

    def __init__(self, hidden_size, latent_size, depth, max_nb, encoding_method="average"):
        super(VAE, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.max_nb = max_nb
        self.encoding_method = encoding_method

        self.encoder = Encoder(self.hidden_size, self.latent_size, self.depth, self.encoding_method)
        self.decoder = Decoder(self.hidden_size, self.latent_size, self.max_nb)

    
    def forward(self, tree_data, encoding_holder, beta, alpha, gamma):
        
        tree_vectors, tree_messages = self.encoder.encode(encoding_holder)
        latent_space_tree_vecs, kl_divergence = self.encoder.rsample(tree_vectors)
        
        pred_loss, stop_loss, pred_acc, stop_acc = self.decoder.forward(tree_data, latent_space_tree_vecs)
        total_loss = pred_loss + stop_loss + beta * kl_divergence

        return total_loss, kl_divergence, pred_acc, stop_acc, pred_loss

    