import torch  # PyTorch tensor library
from torch.autograd import Variable  # Wrapper for automatic differentiation (legacy)
import torch.nn as nn  # Neural network layers and functions
import numpy as np  # Numerical computing library
import torch.nn.functional as F

torch.manual_seed(42)
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
    N_JOINT = 14  # Maximum number of joints (used elsewhere in the codebase), ÄÄÄ brauchen wir nicht

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
    node_batch = []  # Flat list of all nodes from all graphs #ÄÄÄ Liste an Listen oder? Jede Liste ist ein Graph
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
            mess_dict[(x.idx,y.idx)] = len(messages) #ÄÄÄ was macht diese Line, was soll x.idx sein?? !!!
            messages.append( (x,y) )

    # mess_dict contains the node idxs which exchange messages as key and an idx as "message_id" which is used below
    # messages is a list of tuples where each tuple represents a directed message from x to y

    node_graph = [[] for i in range(len(node_batch))] # ÄÄÄ List of lists, each list contains the incoming messages for each node, Which messages each node receives
    mess_graph = [[] for i in range(len(messages))] #ÄÄÄ List of lists, Which messages each message should consider
    fmess = [0] * len(messages) #ÄÄÄ Source node (indices ?) for each message, initialized to 0


    for x,y in messages[1:]:  # Process each message from node x to node y
        mid1 = mess_dict[(x.idx,y.idx)]  # Get message ID for this x→y message
        fmess[mid1] = x.idx  # Record that message mid1 originates from node x
        node_graph[y.idx].append(mid1)  # Tell node y that it receives message mid1
        for z in y.neighbors:  # For each neighbor z of node y
            if z.idx == x.idx: continue  # Skip the sender node x
            mid2 = mess_dict[(y.idx,z.idx)]  # Get message ID for y→z message
            mess_graph[mid2].append(mid1)  # Record that message mid2 (y→z) should consider message mid1 (x→y)
            #ÄÄÄ wieso gehen wir hier nochmal durch die Nachbarn von y?

#Padding the node_graph and mess_graph to ensure they have the same length
    max_len = max([len(t) for t in node_graph] + [1]) #Berechnet die längste Liste, +[1] stellt sicher, dass es mindestens 1 ist, selbst wenn node_graph leer ist (sonst Error)
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
    fnode = torch.LongTensor(np.array(fnode)) #ÄÄÄ Ohne np.array geht es nicht, kriege Fehlermeldung
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

#curr_conn = np.load("data/robot_graphs/adj.npy", allow_pickle=True)
#curr_attr = np.load("data/robot_graphs/feat.npy", allow_pickle=True)



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
#ÄÄÄ Was babbelt der hier über mir
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

def rsample(z_vecs, W_mean, W_var): #W_mean and W_var are nn.Linear layers
    batch_size = z_vecs.size(0)
    z_mean = W_mean(z_vecs) # Apply linear Layer to compute mean of latent Gaussian distribution, z_mean shape is [batch_size, LATENT_SIZE]
    #Computes log variance, then makes it negative to ensure \sigma^2 <= 1 (log(\sigma^2) <= 0)
    z_log_var = -torch.abs(W_var(z_vecs)) #Following Mueller et al.
    # KL = -0.5 * sum(1 + log(\sigma^2) - mean^2 - \sigma^2) / batch_size
    kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size # Dividing by batch_size gives average over batch
    epsilon = create_var_float(torch.randn_like(z_mean)) # epsilon ~ N(0,I)
    # \sigma = exp(0.5 * log(\sigma^2))
    z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon # z_vecs = \mu + \sigma * \epsilon
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

    # vae_train.py -> forward() #ÄÄÄ jtnn_enc.py -> forward() oder vae_train.py -> encode()? 
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

    #ÄÄÄ Ab hier wird es deutlich anders bzw. wir haben Encoding "root" weggelassen
    batch_vecs = []
    for leaf in leafs: # Eine Liste von Listen, wobei jede innere Liste Indizes von Knoten enthält, die zu einem „Blatt-Cluster“ gehören. leafs = [[2, 3], [5], [7, 8, 9]] „dieser Teilbaum hat die Blätter (2,3)“, „dieser nur (5)“, etc.
        cur_vecs = torch.zeros_like(node_vecs[0]) #node_vecs is a tensor of shape [n_nodes, HIDDEN_SIZE]??? Eine Liste (oder Tensor) von Feature-Vektoren pro Knoten.
        for node_idx in leaf:
            cur_vecs += node_vecs[node_idx] #Für jeden „Blatt-Cluster“ in leafs: starte mit einem Nullvektor cur_vecs; summiere die Vektoren aller Knoten in diesem Blatt-Cluster auf
        if ENCODING_METHOD == "average":
            cur_vecs /= len(leaf)
        elif ENCODING_METHOD != "sum":
            exit(f"Encoding method is not in the list")
        batch_vecs.append(cur_vecs)

    tree_vecs = torch.stack(batch_vecs, dim=0)
    return tree_vecs, messages #tree_vecs ist output vom Encoder




T_mean = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)
T_Var = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)



#=== TODO: FROM here we need to continue
#=============== Helper functions for the decoder ===============
feature_dim = 3 # Should be the number of features per node, e.g. 3 for [1, 5, 6]
MAX_NB = 4  # Maximum number of neighbors per node, can be adjusted based on the dataset
#GRU Weights
W_z = nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE)
U_r = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
W_r = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
W_h = nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE)

W = nn.Linear(HIDDEN_SIZE + LATENT_SIZE, HIDDEN_SIZE)

#Stop Prediciton Weights
U = nn.Linear(HIDDEN_SIZE + LATENT_SIZE, HIDDEN_SIZE)
U_i = nn.Linear(2 * HIDDEN_SIZE, HIDDEN_SIZE)

#Output Weights
W_o = nn.Linear(HIDDEN_SIZE, feature_dim)  # Output layer for clique prediction
U_o = nn.Linear(HIDDEN_SIZE, 1)  # Output layer for stop prediction

#features_to_dim = nn.Linear(feature_dim, HIDDEN_SIZE)  #ÄÄÄ Wieso verändert das das Ergebnis ??? Aufeinmal 6 Nodes

#Loss functions
pred_loss_nn = nn.MSELoss(reduction='sum')
stop_loss_nn = nn.BCEWithLogitsLoss(reduction='sum')

def aggregate(hiddens, contexts, x_tree_vecs, mode):
    if mode == 'features': # Renamed from 'word'
        V, V_o = W, W_o
    elif mode == 'stop':
        V, V_o = U, U_o
    else:
        raise ValueError('aggregate mode is wrong')

    tree_contexts = x_tree_vecs.index_select(0, contexts)
    input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
    output_vec = F.relu( V(input_vec) )
    return V_o(output_vec)


#=========== DECODER FORWARD ============
# mol_batch - List of ModTrees in batch,
# x_tree_vecs - encoded tree representation (from latent space)
def decoder_forward(mol_batch, x_tree_vecs):
    pred_hiddens,pred_contexts,pred_targets = [],[],[]
    stop_hiddens,stop_contexts,stop_targets = [],[],[]
    traces = []
    # Generate DFS traces for each tree in the batch
    for mol_tree in mol_batch:
        s = []
        dfs(s, mol_tree.nodes[0], -1)
        traces.append(s)
        for node in mol_tree.nodes:
            node.neighbors = []

    #Predict Root
    batch_size = len(mol_batch)
    pred_hiddens.append(create_var_int(torch.zeros(len(mol_batch),HIDDEN_SIZE))) # Initial hidden states are zeros (no previous context)
    pred_targets.extend([mol_tree.nodes[0].features for mol_tree in mol_batch]) #Actual features of root nodes
    pred_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) ) # Maps each prediction to its corresponding tree in the batch

    max_iter = max([len(tr) for tr in traces]) # Maximum number of steps needed (longest trace in batch)
    padding = create_var_int(torch.zeros(HIDDEN_SIZE), False) # Zero vector used for padding when nodes have fewer neighbors
    h = {} #Dictionary to store hidden states between node pairs: h[(from_node, to_node)]

    # Create a batch of input
    # All the neighbors are set to [] initially
    # Max_nb: max neighbor
    for t in range(max_iter):
        prop_list = []
        batch_list = []
        #Collect active nodes at this step
        for i,plist in enumerate(traces):
            if t < len(plist):
                prop_list.append(plist[t])
                batch_list.append(i) # Keep track of which tree each active node belongs to

        cur_x = []
        cur_h_nei,cur_o_nei = [],[]

        #Process each active node
        for node_x, real_y, _ in prop_list:
            #Neighbors for message passing (target not included)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx] #For each active node, collect neighbor hidden states
            pad_len = MAX_NB - len(cur_nei) # Pad to fixed size MAX_NB (maximum neighbors) for batch processing
            cur_h_nei.extend(cur_nei)
            cur_h_nei.extend([padding] * pad_len)

            #Neighbors for stop prediction (all neighbors)
            cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors] #Stop prediction: Include ALL neighbors (different from message passing)
            pad_len = MAX_NB - len(cur_nei) # Was soll MAX_NB sein !!!
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

            #Current clique embedding
            cur_x.append(node_x.features) #Node features: Collect current node's features

        # Convert features to tensor (no embedding needed)
        #cur_x = torch.tensor(cur_x, dtype=torch.float32)
        #if torch.cuda.is_available():
        #    cur_x = cur_x.cuda()
        #Clique embedding
        cur_x = create_var_int(torch.FloatTensor(np.array(cur_x)))
        cur_x = cur_x  # Skip embedding for now 
        
        #Message passing
        cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE)
        new_h = GRU(cur_x, cur_h_nei, W_z, W_r, U_r, W_h) #GRU message passing: Update hidden states using GRU with neighbor information

        #Node Aggregate
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE) # Was soll MAX_NB sein !!!
        cur_o = cur_o_nei.sum(dim=1) #Aggregation: Sum neighbor hidden states for stop prediction

        #Gather targets
        pred_target,pred_list = [],[]
        stop_target = []
        for i,m in enumerate(prop_list):
            node_x,node_y,direction = m
            x,y = node_x.idx,node_y.idx
            h[(x,y)] = new_h[i] #Hidden state storage: Store computed hidden states for later use
            node_y.neighbors.append(node_x) #Neighbor building: Incrementally build neighbor connections
            if direction == 1: # If this is a forward step in DFS
                pred_target.append(node_y.features) # Target Features: Collect features that need to be predicted
                pred_list.append(i) # Which nodes need prediction
            stop_target.append(direction) #Stop targets: Collect stop decisions (1=continue, 0=stop)

        #Hidden states for stop prediction
        cur_batch = create_var_int(torch.LongTensor(batch_list))
        #stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1)
        stop_hidden = torch.cat([cur_x, cur_o], dim=1) #Woher stammt unsqueeze(0)? !!!
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( cur_batch )
        stop_targets.extend( stop_target ) #Stop data: Store hidden states, contexts, and targets for stop prediction
        
        #Hidden states for clique prediction
        if len(pred_list) > 0:
            batch_list = [batch_list[i] for i in pred_list]
            cur_batch = create_var_int(torch.LongTensor(batch_list))
            pred_contexts.append( cur_batch )
            #Prediction data: Store hidden states, contexts, and targets for feature prediction
            cur_pred = create_var_int(torch.LongTensor(pred_list))
            pred_hiddens.append( new_h.index_select(0, cur_pred) )
            pred_targets.extend( pred_target )

    #Last stop at root
    cur_x,cur_o_nei = [],[]
    for mol_tree in mol_batch:
        node_x = mol_tree.nodes[0]
        cur_x.append(node_x.features) 
        cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
        pad_len = MAX_NB - len(cur_nei) # Was soll MAX_NB sein !!!
        cur_o_nei.extend(cur_nei)
        cur_o_nei.extend([padding] * pad_len)

    
    # Convert features to tensor (no embedding needed)
    #cur_x = torch.tensor(cur_x, dtype=torch.float32)
    #if torch.cuda.is_available():
    #    cur_x = cur_x.cuda()
    cur_x = create_var_int(torch.LongTensor(cur_x))
    cur_x = cur_x # Embedding skipped for now

    cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,HIDDEN_SIZE) # Was soll MAX_NB sein !!!
    cur_o = cur_o_nei.sum(dim=1)

    stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1) #ÄÄÄ Wieder unsqueeze
    stop_hiddens.append( stop_hidden )
    stop_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) ) #ÄÄÄ Wieder create_var_int
    stop_targets.extend( [0] * len(mol_batch) )

    #Predict next clique
    pred_contexts = torch.cat(pred_contexts, dim=0)
    pred_hiddens = torch.cat(pred_hiddens, dim=0)
    pred_scores = aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'features')  # Feature prediction: Use aggregate function to predict node features
    
    # Convert pred_targets to tensor for regression
    #pred_targets = torch.tensor(pred_targets, dtype=torch.float32)
    pred_targets = create_var_int(torch.FloatTensor(pred_targets)) #was LongTensor before


    pred_loss = pred_loss_nn(pred_scores, pred_targets) / len(mol_batch) #Loss calculation: Compute regression loss for feature prediction

    distnace_threshold = 10  # Define a threshold for distance
    distances = torch.norm(pred_scores - pred_targets, dim=1)  # Calculate distances
    pred_acc = torch.mean((distances < distnace_threshold).float())  # Calculate accuracy based on distance threshold, The percentage of predicted node features that are "close enough" to the true node features (within a distance threshold).
    print(f"pred_scores shape: {pred_scores.shape}, pred_targets shape: {pred_targets.shape}")
    print(f"Average prediction distance: {torch.mean(distances).item():.4f}")
    """
    _,preds = torch.max(pred_scores, dim=1)
    print(f"preds shape: {preds.shape}, pred_targets shape: {pred_targets.shape}")
    pred_acc = torch.eq(preds, pred_targets).float()
    pred_acc = torch.sum(pred_acc) / pred_targets.nelement()
    """

    #Predict stop
    stop_contexts = torch.cat(stop_contexts, dim=0)
    stop_hiddens = torch.cat(stop_hiddens, dim=0)
    stop_hiddens = F.relu(U_i(stop_hiddens) )
    stop_scores = aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop') #Stop prediction: Use aggregate function to predict stop decisions
    stop_scores = stop_scores.squeeze(-1)  # Simplified placeholder
    stop_targets = create_var_float(torch.Tensor(stop_targets))
    
    stop_loss = stop_loss_nn(stop_scores, stop_targets) / len(mol_batch)  
    stops = torch.ge(stop_scores, 0).float() #checks if each score is ≥ 0; If score ≥ 0 → decision is 1 (continue expanding) , If score < 0 → decision is 0 (stop expanding)
    stop_acc = torch.eq(stops, stop_targets).float() #compares each prediction with the correct answer; .float() converts to 1.0 (correct) or 0.0 (incorrect)
    stop_acc = torch.sum(stop_acc) / stop_targets.nelement() #The percentage of nodes where the model correctly predicted whether to stop or continue expanding that branch.

    return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

#word_loss, topo_loss, word_acc, topo_acc = decode(x_batch, z_tree_vecs)
MAX_DECODE_LEN = 100
def decoder_decode(x_tree_vecs, prob_decode):
        #assert x_tree_vecs.size(0) == 1 wichtig

        stack = []
        init_hiddens = create_var_int( torch.zeros(1, HIDDEN_SIZE) )
        zero_pad = create_var_int(torch.zeros(1,1,HIDDEN_SIZE))
        contexts = create_var_int( torch.LongTensor(1).zero_() ) #!!! Macht der Zero Vector hier sinn?

        #Root Prediction
        root_features = aggregate(init_hiddens, contexts, x_tree_vecs, 'features')
        #_,root_wid = torch.max(root_score, dim=1) apparently this is not needed, as it is for classification
        #root_wid = root_wid.item() not needed as we have attribute features

        root = TreeNode(root_features) #root = TreeNode(root_features.squeeze().detach()) ???
        #root.wid = root_wid
        root.idx = 0
        stack.append( (root, None) )

        all_nodes = [root]
        h = {}
        for step in range(MAX_DECODE_LEN):
            node_x, _ = stack[-1]
            cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
            if len(cur_h_nei) > 0:
                cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1, HIDDEN_SIZE)
            else:
                cur_h_nei = zero_pad

            #cur_x = create_var_int(torch.LongTensor([node_x.features]))  wichtig
            #cur_x = cur_x  # Skip embedding for now
            #cur_x = torch.tensor(node_x.features, dtype=torch.float32)#.unsqueeze(0)
            #cur_x = cur_x.squeeze(0)
            # if torch.cuda.is_available():
            #    cur_x = cur_x.cuda()
                    # Convert node features to tensor and project to hidden size
            if isinstance(node_x.features, torch.Tensor):
                cur_x = node_x.features.clone().detach()
                if cur_x.dim() == 1:
                    cur_x = cur_x.unsqueeze(0)
            else:
                cur_x = torch.tensor(node_x.features, dtype=torch.float32).unsqueeze(0)

            
            #Predict stop
            cur_h = cur_h_nei.sum(dim=1)
            print(f"Step {step}: cur_x shape: {cur_x.shape}, cur_h shape: {cur_h.shape}")
            # Debug: Print shapes to identify the exact issue
            print(f"cur_x shape: {cur_x.shape}")
            print(f"cur_h shape: {cur_h.shape}")
            print(f"cur_h_nei shape: {cur_h_nei.shape}")
            stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
            stop_hiddens = F.relu(U_i(stop_hiddens) )
            stop_score = aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop')
            print(f"Step {step}: stop_score = {stop_score.item()}")
            if prob_decode:
                backtrack = (torch.bernoulli( torch.sigmoid(stop_score) ).item() == 0)
            else:
                backtrack = (stop_score.item() < 0)
                # print(f'step = {step}, backtrack = {backtrack}, stopscore = {stop_score}')
            print(f"Step {step}: Normal logic, backtrack = {backtrack}")
            if not backtrack: #Forward: Predict next clique
                print(f"Step {step}: Moving forward, predicting new node")
                new_h = GRU(cur_x, cur_h_nei, W_z, W_r, U_r, W_h)
                pred_features = aggregate(new_h, contexts, x_tree_vecs, 'features')
                """
                pred_score = aggregate(new_h, contexts, x_tree_vecs, 'features')
                type_range = VOCAB_SIZE #originally 5
                if prob_decode:
                    sort_wid = torch.multinomial(F.softmax(pred_score, dim=1).squeeze(), type_range)
                else:
                    _,sort_wid = torch.sort(pred_score, dim=1, descending=True)
                    sort_wid = sort_wid.data.squeeze()

                next_wid = None
                for wid in sort_wid[:type_range]:
                    next_wid = wid
                    break

                if next_wid is None:
                    backtrack = True #No more children can be added
                else:
                    node_y = TreeNode(next_wid)
                    node_y.wid = next_wid
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y, None) )
                    all_nodes.append(node_y)
                    """
                # For regression, we directly use the predicted features
                # No need for vocabulary sampling or sorting
                predicted_features = pred_features.squeeze().detach()  # Get the predicted features for the next node
                # Optional: Add some validation or constraints on the predicted features
                # For example, clamp values to reasonable ranges
                # predicted_features = torch.clamp(predicted_features, min=0.0, max=10.0)
                node_y = TreeNode(predicted_features)
                node_y.idx = len(all_nodes)
                node_y.neighbors.append(node_x)
                node_x.neighbors.append(node_y)  #ÄÄÄ später hinzugefügt
                h[(node_x.idx,node_y.idx)] = new_h[0]
                stack.append( (node_y, None) )
                all_nodes.append(node_y)
                print(f"Step {step}: Created new node {node_y.idx} with features {predicted_features}")

            if backtrack: #Backtrack, use if instead of else
                print(f"Step {step}: Backtracking")
                if len(stack) == 1:
                    print(f"Step {step}: At root, terminating")
                    break #At root, terminate

                node_fa,_ = stack[-2]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,HIDDEN_SIZE)
                else:
                    cur_h_nei = zero_pad

                new_h = GRU(cur_x, cur_h_nei, W_z, W_r, U_r, W_h)
                h[(node_x.idx,node_fa.idx)] = new_h[0]
                node_fa.neighbors.append(node_x)
                stack.pop()
                print(f"Step {step}: Popped from stack, now at node {stack[-1][0].idx}")

        return root, all_nodes



print("Shape of current attributes:", np.array(cur_attr).shape)
print("Shape of current connectivity:", np.array(cur_conn).shape)

batch = tensorize(cur_attr, cur_conn)

tree_batch, jtenc_holder = batch

res = encode(jtenc_holder)
tree_vecs = res[0]
messages = res[1]
#print("Encoded tree vectors shape:", tree_vecs.shape)


z_tree_vecs, kl_div = rsample(z_vecs=tree_vecs, W_mean=T_mean, W_var=T_Var)
print("tree_batch: ", [tree_batch[i].nodes for i in range(len(tree_batch))])
print("Encoded tree vectors shape:", z_tree_vecs.shape)
print(z_tree_vecs)

pls = decoder_decode(z_tree_vecs, prob_decode=True)  # prob_decode=False for greedy decoding
#pred_loss, stop_loss, pred_acc, stop_acc = pls

print("Decoded tree structure:")
decoded_tree, all_nodes = pls
print(f"Decoded tree root: {decoded_tree.features}")
print(f"Number of nodes in decoded tree: {len(all_nodes)}")
print("All nodes in decoded tree:")
print(pls)


asd = decoder_forward(tree_batch, z_tree_vecs)
pred_loss, stop_loss, pred_acc, stop_acc = asd
print(f"Prediction Loss: {pred_loss.item()}, Stop Loss: {stop_loss.item()}")
print(f"Prediction Accuracy: {pred_acc}, Stop Accuracy: {stop_acc}")

