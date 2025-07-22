import torch 
from torch.autograd import Variable 
import torch.nn as nn 
import numpy as np

from helper import ModTree, create_var_float, create_var_int, index_select_ND, GraphGRU


HIDDEN_SIZE = 8
LATENT_SIZE = 28
DEPTHT = 3
ENCODING_METHOD = "average"

torch.manual_seed(42)

class Encoder(nn.Module):

    def __init__(self, hidden_size, latent_size, depth, encoding_method, seed=None):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.encoding_method = encoding_method
        

    def encode(self, encoding_holder):

        fnode, fmess, node_graph, mess_graph, scope, leafs = encoding_holder

        
        fnode = create_var_float(fnode)
        fmess = create_var_int(fmess)
        node_graph = create_var_int(node_graph)
        mess_graph = create_var_int(mess_graph)
        messages = create_var_float(torch.zeros(mess_graph.size(0), self.hidden_size))

        fnode = fnode 
        fmess = index_select_ND(fnode, 0, fmess)
        gru = GraphGRU(fnode.shape[1], self.hidden_size, self.depth)
        messages = gru.forward(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=1)
        outputNN = nn.Sequential(nn.Linear((fnode.shape[1] + self.hidden_size), self.hidden_size), nn.ReLU())
        node_vecs = outputNN(node_vecs)
        
        batch_vecs = []
        for leaf in leafs: 
            cur_vecs = torch.zeros_like(node_vecs[0]) 
            for node_idx in leaf:
                cur_vecs += node_vecs[node_idx] 
            if self.encoding_method == "average":
                cur_vecs /= len(leaf)
            elif self.encoding_method != "sum":
                exit(f"Encoding method is not in the list")
            batch_vecs.append(cur_vecs)

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages 
    

    def rsample(self, z_vecs, mean_network, var_network): 

        batch_size = z_vecs.size(0)
        z_mean = mean_network(z_vecs) 
        
        z_log_var = -torch.abs(var_network(z_vecs)) 
        
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size 
        epsilon = create_var_float(torch.randn_like(z_mean)) 
        
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon 
        return z_vecs, kl_loss
        


