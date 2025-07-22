import torch
from torch.autograd import Variable
import torch.nn as nn 
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TreeNode(object):  

    def __init__(self, attr): 

        self.neighbors = []  
        self.features = attr 
        
    def add_neighbor(self, nei_node):  
        self.neighbors.append(nei_node)  

class ModTree(object):  

    def __init__(self, attr, conn):  
        self.nodes = []  
        for i,c in enumerate(attr):  
            node = TreeNode(c)  
            self.nodes.append(node)  

        for i in range(len(attr)): 
            for j in range(i + 1, len(attr)): 
                if conn[i, j] != 0: 
                    self.nodes[i].add_neighbor(self.nodes[j])  
                    self.nodes[j].add_neighbor(self.nodes[i]) 


        for i,node in enumerate(self.nodes): 
            node.nid = i + 1 
            node.is_leaf = (len(node.neighbors) == 1)  

    def size(self): 
        return len(self.nodes) 


def tensorize(attr, conn): 
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


def GRU(x, h_nei, W_z, W_r, U_r, W_h):
    hidden_size = x.size()[-1]
    sum_h = h_nei.sum(dim=1)
    z_input = torch.cat([x,sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))

    r_1 = W_r(x).view(-1,1,hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)
    
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x,sum_gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    new_h = (1.0 - z) * sum_h + z * pre_h
    return new_h

def create_var_float(tensor, requires_grad=False):
    return Variable(tensor.float().to(device), requires_grad=requires_grad)

def create_var_int(tensor, requires_grad=False):
    return Variable(tensor.to(device), requires_grad=requires_grad)

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


class GraphGRU(nn.Module):

    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h, x, mess_graph):
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0 
        mask = create_var_int(mask)
        for it in range(self.depth):
            h_nei = index_select_ND(h, 0, mess_graph)
            sum_h = h_nei.sum(dim=1)
            z_input = torch.cat([x, sum_h], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)
            r_2 = self.U_r(h_nei)
            r = torch.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask

        return h