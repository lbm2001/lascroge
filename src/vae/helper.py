import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable  

from mod_tree import ModTree, TreeNode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_var_float(tensor, requires_grad=False):
    return Variable(tensor.float().to(device), requires_grad=requires_grad)


def create_var_int(tensor, requires_grad=False):
    return Variable(tensor.to(device), requires_grad=requires_grad)


def dfs(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        dfs(stack, y, x.idx)
        stack.append( (y,x,0) )


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

def tree_to_adjacency(tree_root):
    """
    Convert decoded TreeNode back to adjacency matrix by BFS.
    """
    # BFS to assign indices
    node_to_idx = {}
    nodes = []
    queue = [tree_root]
    idx = 0
    # BFS to index every node
    while queue:
        node = queue.pop(0)
        if node not in node_to_idx:
            node_to_idx[node] = idx
            nodes.append(node)
            idx += 1
            for neighbor in node.neighbors:
                if neighbor not in node_to_idx:
                    queue.append(neighbor)
    
    num_nodes = len(nodes) # Wenn Input Shape = Output Shape sein soll, dann hier num_nodes entfernen und als Parameter in die Methode übergeben
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for node, i in node_to_idx.items():
        for neighbor in node.neighbors:
            j = node_to_idx[neighbor]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1
    
    return adj_matrix