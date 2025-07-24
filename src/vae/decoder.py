'''
Modified based on https://github.com/wengong-jin/icml18-jtnn.git.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import create_var_int, create_var_float, depth_first_search, GRU, ModTree, TreeNode

MAX_NB = 4
MAX_DECODE_LEN = 100


class NodePredictionData(object):

    def __init__(self):
        self.hidden_states = []
        self.targets = []
        self.contexts = []

class StopPredictionData(object):

    def __init__(self):
        self.hidden_states = []
        self.targets = []
        self.contexts = []


class DFSHandler(object):

    def __init__(self):
        self.traces = []


    def get_traces(self, trees):
        for tree in trees:
            dfs_stack = []
            self.run_dfs(dfs_stack, tree.nodes[0], -1)
            self.traces.append(dfs_stack)
            for node in tree.nodes:
                node.neighbors = []
        return self.traces


    def run_dfs(stack, x, fa_idx):
        for y in x.neighbors:
            if y.idx == fa_idx: continue
            stack.append( (x,y,1) )
            depth_first_search(stack, y, x.idx)
            stack.append( (y,x,0) )


class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        #GRU Weights
        self.update_gate_linear = nn.Linear(5 + self.hidden_size, hidden_size) # TODO: Change sizes here
        self.reset_gate_neighbor_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.reset_gate_input_linear = nn.Linear(5, hidden_size) # TODO: Change sizes here
        self.candidate_hidden_linear = nn.Linear(2 * hidden_size, hidden_size)

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


    def _generate_dfs_traces(self, tree_data):
        traces = []
        for tree in tree_data:
                dfs_stack = []
                depth_first_search(dfs_stack, tree.nodes[0], -1)
                traces.append(dfs_stack)
                for node in tree.nodes:
                    node.neighbors = []
        return traces


    def _process_dfs_step(self, 
                          current_dfs_step_nodes, 
                          hidden_states_directed_edges, 
                          padding,):
        """
        Collects data from neighboring nodes for node prediction, stop prediction as well as the features and stores them in lists.
        """
        node_prediction_neighbor_hidden_states = []
        stop_prediction_neighbor_hidden_states = []
        node_features = []

        for source_node, target_node, _ in current_dfs_step_nodes:

            #Neighbors for message passing (target not included)
            neighbor_hidden_states_wo_target = [hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors if node_y.idx != target_node.idx]
            node_prediction_neighbor_hidden_states.extend(neighbor_hidden_states_wo_target)
            node_prediction_neighbor_hidden_states.extend([padding] * (MAX_NB - len(neighbor_hidden_states_wo_target)))

            #Neighbors for stop prediction (all neighbors)
            neighbor_hidden_states = [hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors]
            stop_prediction_neighbor_hidden_states.extend(neighbor_hidden_states)
            stop_prediction_neighbor_hidden_states.extend([padding] * (MAX_NB - len(neighbor_hidden_states)))

            # Collect node features
            node_features.append(source_node.features)

        return (node_prediction_neighbor_hidden_states, 
                stop_prediction_neighbor_hidden_states, 
                node_features,)


    def _collect_nodes_at_step(self, traces, step_number):
        """Collect all nodes at the given DFS step across all trees"""
        current_step_nodes = []
        tree_indices = []

        for tree_index, trace in enumerate(traces):
            if step_number < len(trace):
                current_step_nodes.append(trace[step_number])
                tree_indices.append(tree_index)

        return current_step_nodes, tree_indices


    def _collect_prediction_targets(self, current_dfs_step_nodes):
            pred_target = []
            pred_list = []
            stop_target = []

            for i, step in enumerate(current_dfs_step_nodes):
                _, target_node, direction = step
                if direction == 1:
                    pred_target.append(target_node.nid)
                    pred_list.append(i) 
                stop_target.append(direction)
            
            return pred_target, pred_list, stop_target


    def _process_dfs_traces(self, traces, tree_data, latent_space_tree_vecs):
        
        # Setup
        # Data for node prediction
        node_prediction_data = NodePredictionData()
        stop_prediction_data = StopPredictionData()
        
        batch_size = len(tree_data)
        
        # Setup for root
        node_prediction_data.hidden_states.append(create_var_int(torch.zeros(len(tree_data),))) # Initial hidden states for root prediction
        node_prediction_data.targets.extend([tree.nodes[0].features for tree in tree_data]) # Root nodes features
        node_prediction_data.contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) ) # Creates batch indices

        longest_trace = max([len(tr) for tr in traces])
        padding = create_var_int(torch.zeros(self.hidden_size), False)
        hidden_states_directed_edges = {}


        for step_number in range(longest_trace): # Max iterations = biggest trace
            
            current_dfs_step_nodes, tree_batch_indices = self._collect_nodes_at_step(traces=traces, step_number=step_number)

            node_prediction_neighbor_hidden_states, stop_prediction_neighbor_hidden_states, node_features = self._process_dfs_step(
                current_dfs_step_nodes=current_dfs_step_nodes,
                hidden_states_directed_edges=hidden_states_directed_edges,
                padding=padding,
            )
            
            # Create var from features
            node_features = create_var_float(torch.LongTensor(node_features))

            #Message passing
            node_prediction_neighbor_hidden_states = torch.stack(node_prediction_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size)
            new_hidden_states = node_prediction_data.hidden_states
            #new_hidden_states = GRU(node_features, node_prediction_neighbor_hidden_states, self.update_gate_linear, self.reset_gate_input_linear, self.reset_gate_neighbor_linear, self.candidate_hidden_linear)

            #Node Aggregate
            stop_prediction_neighbor_hidden_states = torch.stack(stop_prediction_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size).sum(dim=1)

            #Gather targets
            pred_target, pred_list, stop_target = self._collect_prediction_targets(current_dfs_step_nodes=current_dfs_step_nodes)

            # Update graph structure
            for i, step in enumerate(current_dfs_step_nodes):
                source_node, target_node, _ = step
                hidden_states_directed_edges[(source_node.idx ,target_node.idx)] = new_hidden_states[i]
                target_node.neighbors.append(source_node)


            #Hidden states for stop prediction
            cur_batch = create_var_int(torch.LongTensor(tree_batch_indices))
            stop_hidden = torch.cat([node_features, stop_prediction_neighbor_hidden_states], dim=1)
            stop_prediction_data.hidden_states.append( stop_hidden )
            stop_prediction_data.contexts.append( cur_batch )
            stop_prediction_data.targets.extend( stop_target )
            
            #Hidden states for clique prediction
            if len(pred_list) > 0:
                tree_batch_indices = [tree_batch_indices[i] for i in pred_list]
                cur_batch = create_var_int(torch.LongTensor(tree_batch_indices))
                node_prediction_data.contexts.append( cur_batch )

                cur_pred = create_var_int(torch.LongTensor(pred_list))
                node_prediction_data.hidden_states.append( new_hidden_states.index_select(0, cur_pred) )
                node_prediction_data.targets.extend( pred_target )
            
            return  (node_prediction_data,
                     stop_prediction_data, 
                     hidden_states_directed_edges)


    def forward(self, tree_data, latent_space_tree_vecs):        
        #  Traces saves the nodes in the order we visited them during DFS
        traces = self._generate_dfs_traces(tree_data)
        
        node_prediction_data, stop_prediction_data, hidden_states_directed_edges = self._process_dfs_traces(traces=traces, 
                                                                tree_data=tree_data, 
                                                                latent_space_tree_vecs=latent_space_tree_vecs)

        padding = create_var_int(torch.zeros(self.hidden_size), False)
        batch_size = len(tree_data)

        #Last stop at root
        cur_x = []
        cur_o_nei = []
        for mol_tree in tree_data:
            node_x = mol_tree.nodes[0]
            cur_x.append(node_x.nid)
            cur_nei = [hidden_states_directed_edges[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors]
            pad_len = MAX_NB - len(cur_nei)
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        cur_x = create_var_int(torch.LongTensor(cur_x))
        cur_x = cur_x # This was embedding
        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,MAX_NB,self.hidden_size)
        cur_o = cur_o_nei.sum(dim=1)

        stop_prediction_data.hidden_states.append( torch.cat([cur_x,cur_o], dim=1) )
        stop_prediction_data.contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) )
        stop_prediction_data.targets.extend( [0] * len(tree_data) )

        #Predict next clique
        node_prediction_data.contexts = torch.cat(node_prediction_data.contexts, dim=0)
        node_prediction_data.hidden_states = torch.cat(node_prediction_data.hidden_states, dim=0)
        pred_scores = node_prediction_data.hidden_states   # Skip prediction for now - would need aggregate and pred_loss functions -> Simplified placeholder
        node_prediction_data.targets = create_var_int(torch.LongTensor(node_prediction_data.targets))

        pred_loss = torch.tensor(0.0)  # Simplified placeholder
        _,preds = torch.max(pred_scores, dim=1)
        pred_acc = torch.eq(preds, node_prediction_data.targets).float()
        pred_acc = torch.sum(pred_acc) / node_prediction_data.targets.nelement()

        #Predict stop
        stop_prediction_data.contexts = torch.cat(stop_prediction_data.contexts, dim=0)
        stop_prediction_data.hidden_states = torch.cat(stop_prediction_data.hidden_states, dim=0)
        # Skip stop prediction for now - would need U_i, aggregate and stop_loss functions
        stop_scores = stop_prediction_data.hidden_states.mean(dim=1)  # Simplified placeholder
        stop_prediction_data.targets = create_var_float(torch.Tensor(stop_prediction_data.targets))
        
        stop_loss = torch.tensor(0.0)  # Simplified placeholder
        stops = torch.ge(stop_scores, 0).float()
        stop_acc = torch.eq(stops, stop_prediction_data.targets).float()
        stop_acc = torch.sum(stop_acc) / stop_prediction_data.targets.nelement()

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()



