import torch  # PyTorch tensor library
from torch.autograd import Variable  # Wrapper for automatic differentiation (legacy)
import torch.nn as nn  # Neural network layers and functions
import numpy as np  # Numerical computing library

# Mod Tree: Graph structures in Memory before they get converted to tensors
# Assign attributes

class TreeNode(object):  # Represents a single node in the graph

    def __init__(self, attr):  # Initialize node with feature attributes

        self.neighbors = []  # List to store neighboring nodes
        self.features = attr  # Store node features/attributes
        
    def add_neighbor(self, nei_node):  # Add a neighboring node to this node
        self.neighbors.append(nei_node)  # Append neighbor to the list

class ModTree(object):  # Represents a complete graph structure

    def __init__(self, attr, conn):  # Initialize graph with node attributes and connectivity matrix
        self.nodes = []  # List to store all nodes in the graph
        for i,c in enumerate(attr):  # Iterate through each node's attributes
            node = TreeNode(c)  # Create a new node with these attributes
            self.nodes.append(node)  # Add node to the graph's node list

        for i in range(len(attr)):  # Iterate through all possible node pairs
            for j in range(i + 1, len(attr)):  # Only check upper triangle (avoid duplicates)
                if conn[i, j] != 0:  # If there's a connection between nodes i and j
                    self.nodes[i].add_neighbor(self.nodes[j])  # Add j as neighbor of i
                    self.nodes[j].add_neighbor(self.nodes[i])  # Add i as neighbor of j (undirected)


        for i,node in enumerate(self.nodes):  # Iterate through all nodes to set additional properties
            node.nid = i + 1  # Set node ID (1-indexed)
            node.is_leaf = (len(node.neighbors) == 1)  # Mark as leaf if it has only one neighbor

    def size(self):  # Return the number of nodes in the graph
        return len(self.nodes)  # Return length of nodes list
    

# Transform graphs into the input tensors for the Autoencoder
def tensorize(attr, conn):  # Convert graph data into neural network input format

    # params.py -> N_JOINT
    N_JOINT = 14  # Maximum number of joints (used elsewhere in the codebase)

    # datautils.py -> tensorize()
    # Initializes a ModTree for every graph in conn
    tree_batch = []  # List to store all graphs in the batch
    for idx in range(len(attr)):  # Iterate through each graph in the batch
        mod_tree = ModTree(attr[idx], conn[idx])  # Create graph from attributes and connectivity
        tree_batch.append(mod_tree)  # Add graph to batch

    # datautils.py -> set_batch_nodeID()
    # Sets the idx attribute for each node
    tot = 0  # Counter for global node indexing across all graphs
    for mod_tree in tree_batch:  # Iterate through each graph in batch
        for node in mod_tree.nodes:  # Iterate through each node in current graph
            node.idx = tot  # Assign global index to node
            tot += 1  # Increment counter for next node

    # jtnn_enc.py -> tensorize()
    node_batch = []  # Flat list of all nodes from all graphs
    scope = []  # List of (start_index, num_nodes) tuples for each graph
    leaf = []  # List of leaf node indices for each graph
    for tree in tree_batch:  # Iterate through each graph
        scope.append((len(node_batch), len(tree.nodes)))  # Store start index and count for this graph
        # Scope contains the node idxes for the nodes of each tree, i.e. scope[0] contains range of node indices from the first tree
        node_batch.extend(tree.nodes)  # Add all nodes from current graph to flat list

        tree_leaf = []  # List to store leaf nodes for current graph
        # Sets the leaf -> list of lists containing all leafs of all trees
        for node in tree.nodes:  # Iterate through nodes in current graph
            if len(node.neighbors) == 1: #and node.wid // N_JOINT != 1: # Remove the old filter -> we want to just get the leaf nodes here, we could also use another filter later based on the feature vector
                tree_leaf.append(node.idx)  # Add leaf node index to list
        leaf.append(tree_leaf)  # Add leaf list for this graph to main leaf list
    
    # jtnn_enc.py -> tensorize_nodes()
    messages,mess_dict = [None],{}
    fnode = []
    for x in node_batch: # Loops through all nodes
        fnode.append(x.features) # Adds features of the current node
        for y in x.neighbors:
            mess_dict[(x.idx,y.idx)] = len(messages)
            messages.append( (x,y) )

    # mess_dict contains the node idxs which exchange messages as key and an idx as "message_id" which is used below
    # messages is a list of tuples where each tuple represents a directed message from x to y

    node_graph = [[] for i in range(len(node_batch))]
    mess_graph = [[] for i in range(len(messages))]
    fmess = [0] * len(messages)


    for x,y in messages[1:]:  # Process each message from node x to node y
        mid1 = mess_dict[(x.idx,y.idx)]  # Get message ID for this x→y message
        fmess[mid1] = x.idx  # Record that message mid1 originates from node x
        node_graph[y.idx].append(mid1)  # Tell node y that it receives message mid1
        for z in y.neighbors:  # For each neighbor z of node y
            if z.idx == x.idx: continue  # Skip the sender node x
            mid2 = mess_dict[(y.idx,z.idx)]  # Get message ID for y→z message
            mess_graph[mid2].append(mid1)  # Record that message mid2 (y→z) should consider message mid1 (x→y)

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
    return tree_batch, (fnode, fmess, node_graph, mess_graph, scope, leaf)

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

curr_conn = np.load("/Users/lukasmueller/github/lascroge/data/robot_graphs/adj.npy", allow_pickle=True)
curr_attr = np.load("/Users/lukasmueller/github/lascroge/data/robot_graphs/feat.npy", allow_pickle=True)

batch = tensorize(cur_attr, cur_conn)

# Batch now contains:
#  1. tree_batch: A list of ModTree objects representing the graph structures
#  2. jtenc_holder: A tuple containing tensors for the JTNN encoder:
#    - fnode: Node features tensor [n_nodes, feature_dim]
#    - fmess: Source node indices for each message [n_messages]
#    - node_graph: Incoming message IDs for each node [n_nodes, max_incoming]
#    - mess_graph: Neighboring message IDs for each message [n_messages, max_neighbors]
#    - scope: List of (start_idx, num_nodes) tuples for each graph in batch
#    - leaf: List of leaf node indices for each graph

explanation = """
 1. tree_batch

  [<__main__.ModTree object at 0x108a7a510>, <__main__.ModTree object at 0x108dcd090>]
  - Two ModTree objects representing the two graphs in memory

  2. jtenc_holder tuple contains:

  fnode (Node features)

  tensor([[1, 5, 6],  # Node 0 (graph 1)
          [1, 3, 4],  # Node 1 (graph 1)
          [0, 2, 4],  # Node 2 (graph 1)
          [2, 7, 8],  # Node 3 (graph 2)
          [0, 1, 2],  # Node 4 (graph 2)
          [3, 5, 1]]) # Node 5 (graph 2)
  - Features for all 6 nodes (3 from each graph)

  fmess (Message source nodes)

  tensor([0, 0, 0, 1, 2, 3, 3, 4, 5])
  - Message 0: padding (ignored)
  - Messages 1-2: from node 0
  - Message 3: from node 1
  - Message 4: from node 2
  - Messages 5-6: from node 3
  - Message 7: from node 4
  - Message 8: from node 5

  node_graph (Incoming messages per node)

  tensor([[3, 4],  # Node 0 receives messages 3, 4
          [1, 0],  # Node 1 receives message 1 (+ padding)
          [2, 0],  # Node 2 receives message 2 (+ padding)
          [7, 8],  # Node 3 receives messages 7, 8
          [5, 0],  # Node 4 receives message 5 (+ padding)
          [6, 0]]) # Node 5 receives message 6 (+ padding)

  mess_graph (Neighboring messages per message)

  tensor([[0],  # Message 0: padding
          [4],  # Message 1 (0→1) considers message 4 (0→2)
          [3],  # Message 2 (0→2) considers message 3 (1→0)
          [0],  # Message 3 (1→0) has no neighbors
          [0],  # Message 4 (2→0) has no neighbors
          [8],  # Message 5 (3→4) considers message 8 (3→5)
          [7],  # Message 6 (3→5) considers message 7 (4→3)
          [0],  # Message 7 (4→3) has no neighbors
          [0]]) # Message 8 (5→3) has no neighbors

  scope (Graph boundaries)

  [(0, 3), (3, 3)]
  - Graph 1: nodes 0-2 (start=0, count=3)
  - Graph 2: nodes 3-5 (start=3, count=3)

  leaf (Leaf nodes per graph)

  [[1, 2], [4, 5]]
  - Graph 1: nodes 1, 2 are leaves (only connected to node 0)
  - Graph 2: nodes 4, 5 are leaves (only connected to node 3)
"""

# Now in the training script, we call: loss, kl_div, wacc, tacc, pred_loss = model(batch, loc_batch, beta, alpha, gamma), which is torch syntactic sugar for model.forward()
# So, lets analyze model.forward() and replicate a the complete forward pass (i.e. encoding)

from torch.autograd import Variable

# CONSTANTS (attributes in the model)
HIDDEN_SIZE = 3
LATENT_SIZE = 28
DEPTHT = 3
ENCODING_METHOD = "average"
MAX_NB = 4

#==================== Helper functions and classes ======================

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
        mask[0] = 0 # first vector is padding
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

outputNN = nn.Sequential(nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU())

def rsample(z_vecs, W_mean, W_var):
    batch_size = z_vecs.size(0)
    z_mean = W_mean(z_vecs)
    z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
    kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
    epsilon = create_var_float(torch.randn_like(z_mean))
    z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
    return z_vecs, kl_loss

def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )

#===============#

def encode(jtenc_holder):
    fnode, fmess, node_graph, mess_graph, scope, leafs = jtenc_holder

    # vae_train.py -> forward()
    fnode = create_var_float(fnode)
    fmess = create_var_int(fmess)
    node_graph = create_var_int(node_graph)
    mess_graph = create_var_int(mess_graph)
    messages = create_var_float(torch.zeros(mess_graph.size(0), HIDDEN_SIZE))

    fnode = fnode # Here we skip the embedding
    fmess = index_select_ND(fnode, 0, fmess)
    gru = GraphGRU(HIDDEN_SIZE, HIDDEN_SIZE, DEPTHT)
    messages = gru.forward(messages, fmess, mess_graph)

    mess_nei = index_select_ND(messages, 0, node_graph)
    node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=1)
    node_vecs = outputNN(node_vecs)
    
    max_len = max([x for _,x in scope])

    batch_vecs = []
    for leaf in leafs:
        cur_vecs = torch.zeros_like(node_vecs[0])
        for node_idx in leaf:
            cur_vecs += node_vecs[node_idx]
        if ENCODING_METHOD == "average":
            cur_vecs /= len(leaf)
        elif ENCODING_METHOD != "sum":
            exit(f"Encoding method is not in the list")
        batch_vecs.append(cur_vecs)

    tree_vecs = torch.stack(batch_vecs, dim=0)
    return tree_vecs, messages


tree_batch, jtenc_holder = batch

res = encode(jtenc_holder)
tree_vecs = res[0]
messages = res[1]

T_mean = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)
T_Var = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)

z_tree_vecs, kl_div = rsample(z_vecs=tree_vecs, W_mean=T_mean, W_var=T_Var)

#=== TODO: FROM here we need to continue
#=========== DECODER FORWARD ============

def decoder_forward(mol_batch, x_tree_vecs):
    pred_hiddens,pred_contexts,pred_targets = [],[],[]
    stop_hiddens,stop_contexts,stop_targets = [],[],[]
    traces = []
    for mol_tree in mol_batch:
        s = []
        dfs(s, mol_tree.nodes[0], -1)
        traces.append(s)
        for node in mol_tree.nodes:
            node.neighbors = []

    #Predict Root
    batch_size = len(mol_batch)
    pred_hiddens.append(create_var_int(torch.zeros(len(mol_batch),HIDDEN_SIZE)))
    pred_targets.extend([mol_tree.nodes[0].nid for mol_tree in mol_batch])
    pred_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) )

    max_iter = max([len(tr) for tr in traces])
    padding = create_var_int(torch.zeros(HIDDEN_SIZE), False)
    h = {}

    # Create a batch of input
    # All the neighbors are set to [] initially
    # Max_nb: max neighbor
    for t in range(max_iter):
        prop_list = []
        batch_list = []
        for i,plist in enumerate(traces):
            if t < len(plist):
                prop_list.append(plist[t])
                batch_list.append(i)

        cur_x = []
        cur_h_nei,cur_o_nei = [],[]

        for node_x, real_y, _ in prop_list:
            #Neighbors for message passing (target not included)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx]
            pad_len = MAX_NB - len(cur_nei)
            cur_h_nei.extend(cur_nei)
            cur_h_nei.extend([padding] * pad_len)

            #Neighbors for stop prediction (all neighbors)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

            #Current clique embedding
            cur_x.append(node_x.nid)

        #Clique embedding
        cur_x = create_var_int(torch.LongTensor(cur_x))
        cur_x = cur_x  # Skip embedding for now 
        
        #Message passing
        cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE)
        # Skip GRU for now - would need to initialize GRU weights
        new_h = cur_x  # Simplified placeholder

        #Node Aggregate
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE)
        cur_o = cur_o_nei.sum(dim=1)

        #Gather targets
        pred_target,pred_list = [],[]
        stop_target = []
        for i,m in enumerate(prop_list):
            node_x,node_y,direction = m
            x,y = node_x.idx,node_y.idx
            h[(x,y)] = new_h[i]
            node_y.neighbors.append(node_x)
            if direction == 1:
                pred_target.append(node_y.nid)
                pred_list.append(i) 
            stop_target.append(direction)

        #Hidden states for stop prediction
        cur_batch = create_var_int(torch.LongTensor(batch_list))
        stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1)
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( cur_batch )
        stop_targets.extend( stop_target )
        
        #Hidden states for clique prediction
        if len(pred_list) > 0:
            batch_list = [batch_list[i] for i in pred_list]
            cur_batch = create_var_int(torch.LongTensor(batch_list))
            pred_contexts.append( cur_batch )

            cur_pred = create_var_int(torch.LongTensor(pred_list))
            pred_hiddens.append( new_h.index_select(0, cur_pred) )
            pred_targets.extend( pred_target )

    #Last stop at root
    cur_x,cur_o_nei = [],[]
    for mol_tree in mol_batch:
        node_x = mol_tree.nodes[0]
        cur_x.append(node_x.nid)
        cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
        pad_len = MAX_NB - len(cur_nei)
        cur_o_nei.extend(cur_nei)
        cur_o_nei.extend([padding] * pad_len)

    cur_x = create_var_int(torch.LongTensor(cur_x))
    cur_x = cur_x
    cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE)
    cur_o = cur_o_nei.sum(dim=1)

    stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1)
    stop_hiddens.append( stop_hidden )
    stop_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) )
    stop_targets.extend( [0] * len(mol_batch) )

    #Predict next clique
    pred_contexts = torch.cat(pred_contexts, dim=0)
    pred_hiddens = torch.cat(pred_hiddens, dim=0)
    # Skip prediction for now - would need aggregate and pred_loss functions
    pred_scores = pred_hiddens  # Simplified placeholder
    pred_targets = create_var_int(torch.LongTensor(pred_targets))

    pred_loss = torch.tensor(0.0)  # Simplified placeholder
    _,preds = torch.max(pred_scores, dim=1)
    pred_acc = torch.eq(preds, pred_targets).float()
    pred_acc = torch.sum(pred_acc) / pred_targets.nelement()

    #Predict stop
    stop_contexts = torch.cat(stop_contexts, dim=0)
    stop_hiddens = torch.cat(stop_hiddens, dim=0)
    # Skip stop prediction for now - would need U_i, aggregate and stop_loss functions
    stop_scores = stop_hiddens.mean(dim=1)  # Simplified placeholder
    stop_targets = create_var_float(torch.Tensor(stop_targets))
    
    stop_loss = torch.tensor(0.0)  # Simplified placeholder
    stops = torch.ge(stop_scores, 0).float()
    stop_acc = torch.eq(stops, stop_targets).float()
    stop_acc = torch.sum(stop_acc) / stop_targets.nelement()

    return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()



#word_loss, topo_loss, word_acc, topo_acc = decode(x_batch, z_tree_vecs)
