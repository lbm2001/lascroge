'''
Modified based on https://github.com/wengong-jin/icml18-jtnn.git.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import create_var_int, create_var_float, GRU, ModTree, TreeNode

MAX_NB = 4
MAX_DECODE_LEN = 100

class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        #GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        #Word Prediction Weights 
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        #Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        #Output Weights
        self.W_o = nn.Linear(hidden_size, 3 * 14)
        self.U_o = nn.Linear(hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'word':
            V, V_o = self.W, self.W_o
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu( V(input_vec) )
        return V_o(output_vec)

    def forward(self, tree_data, latent_space_tree_vecs):
        node_prediction_hidden_states = []
        node_prediction_contexts = [] 
        node_prediction_targets = []
        
        expansion_stop_hidden_states = []
        expansion_stop_contexts = []
        expansion_stop_targets = []
        
        traces = []
        
        # Traces saves the nodes in the order we visited them during DFS
        for tree in tree_data:
            dfs_stack = []
            depth_first_search(dfs_stack, tree.nodes[0], -1)
            traces.append(dfs_stack)
            for node in tree.nodes:
                node.neighbors = []

        #Predict Root
        batch_size = len(tree_data)
        node_prediction_hidden_states.append(create_var_int(torch.zeros(len(tree_data),)))
        node_prediction_targets.extend([mol_tree.nodes[0].features for mol_tree in tree_data])
        node_prediction_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) )

        longest_trace = max([len(tr) for tr in traces])
        max_iterations = longest_trace
        padding = create_var_int(torch.zeros(self.hidden_size), False)
        hidden_states_directed_edges = {}

        # Create a batch of input
        # All the neighbors are set to [] initially
        # Max_nb: max neighbor
        for t in range(max_iterations): # Max iterations = biggest trace
            current_dfs_step_nodes = [] # Nodes being processed in this DFS step
            tree_batch_indices = []
            for i, trace in enumerate(traces):
                if t < len(trace):
                    current_dfs_step_nodes.append(trace[t])
                    tree_batch_indices.append(i)
            # prop list is the traces flattened and ordered by "step number"
            current_node_features = []
            current_neighbor_hidden_states = []
            aggregated_neighbor_hidden_states = []

            # Process each DFS step
            for source_node, target_node, _ in current_dfs_step_nodes:
                # source_node : From where we are going, node_y : Where we are going
                #Neighbors for message passing (target not included)
                neighbor_hidden_states_wo_target = [hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors if node_y.idx != target_node.idx]
                pad_len = MAX_NB - len(neighbor_hidden_states_wo_target)
                current_neighbor_hidden_states.extend(neighbor_hidden_states_wo_target)
                current_neighbor_hidden_states.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                neighbor_hidden_states = [hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors]
                pad_len = MAX_NB - len(neighbor_hidden_states)
                aggregated_neighbor_hidden_states.extend(neighbor_hidden_states)
                aggregated_neighbor_hidden_states.extend([padding] * pad_len)

                #Current clique embedding
                current_node_features.append(source_node.features)

            #Clique embedding
            current_node_features = create_var_int(torch.LongTensor(current_node_features))
            current_node_features = current_node_features  # Skip embedding for now 
            
            #Message passing
            current_neighbor_hidden_states = torch.stack(current_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size)
            # Skip GRU for now - would need to initialize GRU weights
            new_h = current_node_features  # Simplified placeholder

            #Node Aggregate
            aggregated_neighbor_hidden_states = torch.stack(aggregated_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size)
            aggregated_neighbors_states = aggregated_neighbor_hidden_states.sum(dim=1)

            #Gather targets
            pred_target,pred_list = [],[]
            stop_target = []
            for i,m in enumerate(current_dfs_step_nodes):
                source_node,node_y,direction = m
                x,y = source_node.idx,node_y.idx
                hidden_states_directed_edges[(x,y)] = new_h[i]
                node_y.neighbors.append(source_node)
                if direction == 1:
                    pred_target.append(node_y.nid)
                    pred_list.append(i) 
                stop_target.append(direction)

            #Hidden states for stop prediction
            cur_batch = create_var_int(torch.LongTensor(tree_batch_indices))
            stop_hidden = torch.cat([current_node_features.unsqueeze(0) if current_node_features.dim() == 0 else current_node_features, aggregated_neighbors_states], dim=1)
            expansion_stop_hidden_states.append( stop_hidden )
            expansion_stop_contexts.append( cur_batch )
            expansion_stop_targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                tree_batch_indices = [tree_batch_indices[i] for i in pred_list]
                cur_batch = create_var_int(torch.LongTensor(tree_batch_indices))
                node_prediction_contexts.append( cur_batch )

                cur_pred = create_var_int(torch.LongTensor(pred_list))
                node_prediction_hidden_states.append( new_h.index_select(0, cur_pred) )
                node_prediction_targets.extend( pred_target )

        #Last stop at root
        cur_x,cur_o_nei = [],[]
        for mol_tree in tree_data:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.nid)
            cur_nei = [hidden_states_directed_edges[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var_int(torch.LongTensor(cur_x))
        cur_x = cur_x
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1)
        expansion_stop_hidden_states.append( stop_hidden )
        expansion_stop_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) )
        expansion_stop_targets.extend( [0] * len(tree_data) )

        #Predict next clique
        node_prediction_contexts = torch.cat(node_prediction_contexts, dim=0)
        node_prediction_hidden_states = torch.cat(node_prediction_hidden_states, dim=0)
        # Skip prediction for now - would need aggregate and pred_loss functions
        pred_scores = node_prediction_hidden_states  # Simplified placeholder
        node_prediction_targets = create_var_int(torch.LongTensor(node_prediction_targets))

        pred_loss = torch.tensor(0.0)  # Simplified placeholder
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, node_prediction_targets).float()
        pred_acc = torch.sum(pred_acc) / node_prediction_targets.nelement()

        #Predict stop
        expansion_stop_contexts = torch.cat(expansion_stop_contexts, dim=0)
        expansion_stop_hidden_states = torch.cat(expansion_stop_hidden_states, dim=0)
        # Skip stop prediction for now - would need U_i, aggregate and stop_loss functions
        stop_scores = expansion_stop_hidden_states.mean(dim=1)  # Simplified placeholder
        expansion_stop_targets = create_var_float(torch.Tensor(expansion_stop_targets))
        
        stop_loss = torch.tensor(0.0)  # Simplified placeholder
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, expansion_stop_targets).float()
        stop_acc = torch.sum(stop_acc) / expansion_stop_targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()

"""
Helper Functions:
"""
def depth_first_search(stack, x, fa_idx):
    for y in x.neighbors:
        if y.idx == fa_idx: continue
        stack.append( (x,y,1) )
        depth_first_search(stack, y, x.idx)
        stack.append( (y,x,0) )
