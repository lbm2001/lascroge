import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# Mod Tree: Graph structures in Memory before they get converted to tensors
# Assign attributes

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
    

# Transform graphs into the input tensors for the Autoencoder
def tensorize(attr, conn):

    # params.py -> N_JOINT
    N_JOINT = 14

    # datautils.py -> tensorize()
    # Initializes a ModTree for every graph in conn
    tree_batch = []
    for idx in range(len(attr)):
        mod_tree = ModTree(attr[idx], conn[idx])
        tree_batch.append(mod_tree)

    # datautils.py -> set_batch_nodeID()
    tot = 0
    for mod_tree in tree_batch:
        for node in mod_tree.nodes:
            node.idx = tot
            tot += 1

    # jtnn_enc.py -> tensorize()
    node_batch = [] 
    scope = []
    leaf = []
    for tree in tree_batch:
        scope.append((len(node_batch), len(tree.nodes)))
        node_batch.extend(tree.nodes)

        tree_leaf = []
        for node in tree.nodes:
            if len(node.neighbors) == 1: #and node.wid // N_JOINT != 1: # Remove the old filter -> we want to just get the leaf nodes here, we could also use another filter later based on the feature vector
                tree_leaf.append(node.idx)
        leaf.append(tree_leaf)
    
    # jtnn_enc.py -> tensorize_nodes()
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
    fnode = torch.LongTensor(fnode)
    return (fnode, fmess, node_graph, mess_graph, scope, leaf), mess_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    
    
    
    
    

# Encoding Pipeline
# Lets walk through it by an example with two graphs
# Mock the data

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

batch = tensorize(cur_attr, cur_conn)
# Batch now contains:
# batch[0] = tuple of six elements used for encoding a batch of graphs
# batch[0][0] = fnode = real-valued node features from all graphs,
#                shape: [N_nodes_total, feature_dim], e.g. [[1, 5, 6], [1, 3, 4], ...]
#
# batch[0][1] = fmess = list of source node indices for each message (including dummy at index 0),
#                shape: [N_messages], where fmess[i] = index of the source node of message i
#
# batch[0][2] = node_graph = for each node, a list of incoming message indices (padded),
#                shape: [N_nodes_total, max_incoming_messages], e.g. [[3, 4], [1, 0], ...]
#
# batch[0][3] = mess_graph = for each message, a list of neighboring message indices (padded),
#                shape: [N_messages, max_neighbors], used for message-to-message passing in GraphGRU
#
# batch[0][4] = scope = list of (start_idx, num_nodes) for each graph in the batch,
#                used to re-separate flattened node features into individual graphs
#
# batch[0][5] = leaf = list of lists, where each inner list contains indices of leaf nodes
#                in the corresponding graph (used for pooling into latent graph vectors)
#
# batch[1] = mess_dict = a dictionary mapping (source_node_idx, target_node_idx) → message_index,
#             helpful for debugging and reconstructing message flows

# Now in the training script, we call: loss, kl_div, wacc, tacc, pred_loss = model(batch, loc_batch, beta, alpha, gamma), which is torch syntactic sugar for model.forward()
# So, lets analyze model.forward()




