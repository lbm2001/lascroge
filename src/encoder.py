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

class Encoder():

    def __init__(self, hidden_size, latent_size, depth, encoding_method, seed=None):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.encoding_method = encoding_method
        



    def tensorize(self, attr, conn): 

        tree_batch = [] 
        for idx in range(len(attr)): 
            mod_tree = ModTree(attr[idx], conn[idx])  
            tree_batch.append(mod_tree) 

        tot = 0  
        for mod_tree in tree_batch: 
            for node in mod_tree.nodes:  
                node.idx = tot  
                tot += 1  

        
        node_batch = []  
        scope = []  
        leaf = []  
        for tree in tree_batch:  
            scope.append((len(node_batch), len(tree.nodes)))  
            
            node_batch.extend(tree.nodes)  

            tree_leaf = []  
            
            for node in tree.nodes:  
                if len(node.neighbors) == 1: 
                    tree_leaf.append(node.idx)  
            leaf.append(tree_leaf)  
        
        
        messages,mess_dict = [None],{}
        fnode = []
        for x in node_batch: 
            fnode.append(x.features) 
            for y in x.neighbors:
                mess_dict[(x.idx,y.idx)] = len(messages) 
                messages.append( (x,y) )

        
        

        node_graph = [[] for i in range(len(node_batch))] 
        mess_graph = [[] for i in range(len(messages))] 
        fmess = [0] * len(messages) 


        for x,y in messages[1:]:  
            mid1 = mess_dict[(x.idx,y.idx)]  
            fmess[mid1] = x.idx  
            node_graph[y.idx].append(mid1)  
            for z in y.neighbors:  
                if z.idx == x.idx: continue  
                mid2 = mess_dict[(y.idx,z.idx)]  
                mess_graph[mid2].append(mid1)  
                

    
        max_len = max([len(t) for t in node_graph] + [1]) 
        for t in node_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        max_len = max([len(t) for t in mess_graph] + [1])
        for t in mess_graph:
            pad_len = max_len - len(t)
            t.extend([0] * pad_len)

        mess_graph = torch.LongTensor(mess_graph)
        node_graph = torch.LongTensor(node_graph)
        fmess = torch.LongTensor(fmess)
        fnode = torch.LongTensor(np.array(fnode)) 
        return tree_batch, (fnode, fmess, node_graph, mess_graph, scope, leaf)


    def encode(self, attr, conn):

        _, jtenc_holder = self.tensorize(attr, conn)

        fnode, fmess, node_graph, mess_graph, scope, leafs = jtenc_holder

        
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
    

    def rsample(self, z_vecs): 

        W_mean = nn.Linear(self.hidden_size, self.latent_size)
        W_var = nn.Linear(self.hidden_size, self.latent_size)

        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs) 
        
        z_log_var = -torch.abs(W_var(z_vecs)) 
        
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size 
        epsilon = create_var_float(torch.randn_like(z_mean)) 
        
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon 
        return z_vecs, kl_loss
        


# Test
conn = np.load("/Users/lukasmueller/github/lascroge/data/robot_graphs/adj.npy", allow_pickle=True)
attr = np.load("/Users/lukasmueller/github/lascroge/data/robot_graphs/feat.npy", allow_pickle=True)

encoder = Encoder(hidden_size=HIDDEN_SIZE, 
                  latent_size=LATENT_SIZE, 
                  depth=DEPTHT, 
                  encoding_method=ENCODING_METHOD)

x_vecs, _ = encoder.encode(attr, conn)
z_vecs = encoder.rsample(x_vecs)

print(x_vecs)
#print(z_vecs)
