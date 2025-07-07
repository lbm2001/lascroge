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

#==================== Helper functions and classes ======================

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



res = encode(batch[1])
tree_vecs = res[0]
messages = res[1]

T_mean = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)
T_Var = nn.Linear(HIDDEN_SIZE, LATENT_SIZE)

z_tree_vecs, kl_div = rsample(z_vecs=tree_vecs, W_mean=T_mean, W_var=T_Var)

print(z_tree_vecs)

FROM_CLAUDE = """




import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# Helper function for advanced indexing
def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

# GraphGRU: Custom GRU implementation for graph message passing
class GraphGRU(nn.Module):
    def __init__(self, input_size, hidden_size, depth):
        super(GraphGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth  # Number of message passing iterations

        # GRU gates: z = update gate, r = reset gate, h = candidate hidden state
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)  # Update gate weights
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)    # Reset gate weights (input part)
        self.U_r = nn.Linear(hidden_size, hidden_size)               # Reset gate weights (hidden part)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)  # Candidate hidden state weights

    def forward(self, h, x, mess_graph):
        # h: current hidden states for all messages [n_messages, hidden_size]
        # x: input features for all messages [n_messages, hidden_size]
        # mess_graph: adjacency matrix for message-to-message connections [n_messages, max_neighbors]

        # Create mask to ignore padding (first message is always padding)
        mask = torch.ones(h.size(0), 1)
        mask[0] = 0  # First message is padding, so mask it out
        if torch.cuda.is_available():
            mask = mask.cuda()

        # Iterate through message passing steps
        for it in range(self.depth):
            # Gather neighboring hidden states for each message using index_select_ND
            # mess_graph[i] contains indices of neighboring messages for message i
            h_nei = index_select_ND(h, 0, mess_graph)  # [n_messages, max_neighbors, hidden_size]
            sum_h = h_nei.sum(dim=1)  # Sum over neighbors [n_messages, hidden_size]

            # Update gate: decides how much of the new candidate to accept
            z_input = torch.cat([x, sum_h], dim=1)  # Concatenate input and neighbor sum
            z = torch.sigmoid(self.W_z(z_input))    # Update gate values [0,1]

            # Reset gate: decides how much of the previous hidden state to forget
            r_1 = self.W_r(x).view(-1, 1, self.hidden_size)  # Input contribution
            r_2 = self.U_r(h_nei)                            # Hidden state contribution
            r = torch.sigmoid(r_1 + r_2)                     # Reset gate values [0,1]

            # Apply reset gate to neighbor hidden states
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)

            # Candidate hidden state: new information to potentially add
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))

            # Final hidden state: interpolation between old and new
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask  # Apply mask to ignore padding

        return h

# JTNNEncoder: Main encoder class that processes tensorized graphs
class JTNNEncoder(nn.Module):
    def __init__(self, hidden_size, depth, embedding, encoding_method):
        super(JTNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.embedding = embedding  # Embedding layer for node features
        self.encoding_method = encoding_method  # "root", "sum", or "average"

        # Output network: combines node features with aggregated messages
        self.outputNN = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),  # 2x because we concat node + message features
            nn.ReLU()
        )

        # GraphGRU for message passing
        self.GRU = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode, fmess, node_graph, mess_graph, scope, leafs):
        # fnode: node features [n_nodes, feature_dim]
        # fmess: message source node indices [n_messages]
        # node_graph: incoming message indices for each node [n_nodes, max_incoming]
        # mess_graph: neighboring message indices for each message [n_messages, max_neighbors]
        # scope: list of (start_idx, num_nodes) for each graph in batch
        # leafs: list of leaf node indices for each graph

        # Convert to Variables (for older PyTorch versions)
        if torch.cuda.is_available():
            fnode = fnode.cuda()
            fmess = fmess.cuda()
            node_graph = node_graph.cuda()
            mess_graph = mess_graph.cuda()

        # Initialize message hidden states
        messages = torch.zeros(mess_graph.size(0), self.hidden_size)
        if torch.cuda.is_available():
            messages = messages.cuda()

        # Step 1: Embed node features
        fnode = self.embedding(fnode)  # [n_nodes, hidden_size]

        # Step 2: Create initial message features from source nodes
        fmess = index_select_ND(fnode, 0, fmess)  # [n_messages, hidden_size] - gather source node features

        # Step 3: Run message passing with GraphGRU
        messages = self.GRU(messages, fmess, mess_graph)  # [n_messages, hidden_size]

        # Step 4: Aggregate messages for each node
        mess_nei = index_select_ND(messages, 0, node_graph)  # [n_nodes, max_incoming, hidden_size]

        # Step 5: Combine node features with aggregated messages
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=-1)  # [n_nodes, 2*hidden_size]
        node_vecs = self.outputNN(node_vecs)  # [n_nodes, hidden_size]

        # Step 6: Pool node representations into graph-level representations
        batch_vecs = []

        if self.encoding_method == "root":
            # Use the root node (first node) representation for each graph
            for st, le in scope:
                cur_vecs = node_vecs[st]  # Root is the first node
                batch_vecs.append(cur_vecs)
        else:
            # Use leaf nodes for pooling
            for leaf in leafs:
                cur_vecs = torch.zeros_like(node_vecs[0])
                for node_idx in leaf:
                    cur_vecs += node_vecs[node_idx]

                if self.encoding_method == "average":
                    cur_vecs /= len(leaf)  # Average over leaf nodes
                elif self.encoding_method == "sum":
                    pass  # Already summed above
                else:
                    raise ValueError(f"Unknown encoding method: {self.encoding_method}")

                batch_vecs.append(cur_vecs)

        # Stack individual graph representations into batch
        tree_vecs = torch.stack(batch_vecs, dim=0)  # [batch_size, hidden_size]
        return tree_vecs, messages

# JTNNVAE: Complete VAE model with encoder/decoder
class JTNNVAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, latent_size, depth, encoding_method):
        super(JTNNVAE, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder component
        self.jtnn = JTNNEncoder(
            hidden_size=hidden_size,
            depth=depth,
            embedding=nn.Embedding(vocab_size, hidden_size),
            encoding_method=encoding_method
        )

        # Variational layers: map from hidden representation to latent space
        self.T_mean = nn.Linear(hidden_size, latent_size)  # Mean of latent distribution
        self.T_var = nn.Linear(hidden_size, latent_size)   # Log variance of latent distribution

    def encode(self, jtenc_holder):
        # jtenc_holder is the tuple returned by tensorize(): (fnode, fmess, node_graph, mess_graph, scope, leafs)
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        return tree_vecs, tree_mess

    def encode_latent(self, x_batch):
        # Encode to latent space with mean and variance
        jtenc_holder = x_batch[1]  # Assuming x_batch is (data, jtenc_holder)
        tree_vecs, _ = self.jtnn(*jtenc_holder)

        # Compute mean and log variance for variational encoding
        tree_mean = self.T_mean(tree_vecs)
        tree_var = -torch.abs(self.T_var(tree_vecs))  # Negative absolute value for log variance

        return tree_mean, tree_var

    def rsample(self, z_vecs, W_mean, W_var):
        # Reparameterization trick for variational sampling
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs))  # Following Mueller et al.

        # Compute KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0,I)
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size

        # Sample from latent distribution using reparameterization trick
        epsilon = torch.randn_like(z_mean)
        if torch.cuda.is_available():
            epsilon = epsilon.cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon

        return z_vecs, kl_loss





"""