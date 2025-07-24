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


class TracesHandler(object):

    def __init__(self, tree_batch):
        
        self.tree_batch = tree_batch

        dfs_handler = DFSHandler()
        self.traces = dfs_handler.get_traces(tree_batch)

        self.longest_trace = max([len(tr) for tr in self.traces])
        self.batch_size = len(self.tree_batch)
        self.current_batch_list = []

        self.node_prediction_data = NodePredictionData()
        self.stop_prediction_data = StopPredictionData()
        self.hidden_states_directed_edges = {}
    
    def _setup_root_prediction(self):
        self.node_prediction_data.hidden_states.append(create_var_int(torch.zeros(self.batch_size,))) # Initial hidden states for root prediction
        self.node_prediction_data.targets.extend([tree.nodes[0].features for tree in self.tree_batch]) # Root nodes features
        self.node_prediction_data.contexts.append( create_var_int( torch.LongTensor(range(self.batch_size)) ) ) # Creates batch indices
    
    def _collect_steps(self, step_number):
        """Collects all steps at the current DFS step across all trees"""
        current_steps = []
        tree_indices = []

        for tree_index, trace in enumerate(self.traces):
            if step_number < len(trace):
                current_steps.append(trace[step_number])

        return current_steps, tree_indices

    def _process_dfs_step_node_prediction(self, current_dfs_steps, padding, update_gate_linear, reset_gate_input_linear, reset_gate_neighbor_linear, candidate_hidden_linear):
        node_prediction_neighbor_hidden_states = []
        node_features = []

        for source_node, target_node, _ in current_dfs_steps:
            #Neighbors for message passing (target not included)
            neighbor_hidden_states_wo_target = [self.hidden_states_directed_edges[(neighbor.idx, source_node.idx)] for neighbor in source_node.neighbors if neighbor.idx != target_node.idx]
            node_prediction_neighbor_hidden_states.extend(neighbor_hidden_states_wo_target)
            node_prediction_neighbor_hidden_states.extend([padding] * (MAX_NB - len(neighbor_hidden_states_wo_target)))

            node_features.append(source_node.features)
        
        node_prediction_neighbor_hidden_states = torch.stack(node_prediction_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size)
        node_features = create_var_float(torch.LongTensor(node_features))

        new_hidden_states = GRU(node_features, node_prediction_neighbor_hidden_states, update_gate_linear, reset_gate_input_linear, reset_gate_neighbor_linear, candidate_hidden_linear)

        self._update_graph(current_dfs_steps, new_hidden_states)

        node_prediction_targets, node_prediction_list, _ = self._collect_prediction_targets(current_dfs_steps)

        if len(node_prediction_list) > 0:
                batch_list = [self.current_batch_list[i] for i in node_prediction_list]
                self.node_prediction_data.contexts.append( create_var_int(torch.LongTensor(batch_list)) )

                cur_pred = create_var_int(torch.LongTensor(node_prediction_list))
                self.node_prediction_data.hidden_states.append( new_hidden_states.index_select(0, cur_pred) )
                self.node_prediction_data.targets.extend( node_prediction_targets )

    def _process_dfs_step_stop_prediction(self, current_dfs_steps, padding):
        
        stop_prediction_neighbor_hidden_states = []
        node_features = []

        #Neighbors for stop prediction (all neighbors)
        for source_node, _, _ in current_dfs_steps:
            neighbor_hidden_states = [self.hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors]
            stop_prediction_neighbor_hidden_states.extend(neighbor_hidden_states)
            stop_prediction_neighbor_hidden_states.extend([padding] * (MAX_NB - len(neighbor_hidden_states)))

            node_features.append(source_node.features)
            
        # Node aggregate
        stop_prediction_neighbor_hidden_states = torch.stack(stop_prediction_neighbor_hidden_states, dim=0).view(-1,MAX_NB,self.hidden_size).sum(dim=1)
        node_features = create_var_float(torch.LongTensor(node_features))

        _, _, stop_target = self._collect_prediction_targets(current_dfs_steps)

        #Hidden states for stop prediction
        cur_batch = create_var_int(torch.LongTensor(self.current_batch_list))
        stop_hidden = torch.cat([node_features, stop_prediction_neighbor_hidden_states], dim=1)
        self.stop_prediction_data.hidden_states.append( stop_hidden )
        self.stop_prediction_data.contexts.append( cur_batch )
        self.stop_prediction_data.targets.extend( stop_target )

    def _collect_prediction_targets(self, current_dfs_steps):
        node_prediction_targets = []
        node_prediction_list = []
        stop_target = []

        for i, step in enumerate(current_dfs_steps):
            _, target_node, direction = step
            if direction == 1:
                node_prediction_targets.append(target_node.nid)
                node_prediction_list.append(i) 
            stop_target.append(direction)
        
        return node_prediction_targets, node_prediction_list, stop_target

    def _update_graph(self, current_dfs_steps, new_hidden_states):
        for i, step in enumerate(current_dfs_steps):
            source_node, target_node, _ = step
            self.hidden_states_directed_edges[(source_node.idx ,target_node.idx)] = new_hidden_states[i]
            target_node.neighbors.append(source_node)

    def process(self, hidden_size, update_gate_linear, reset_gate_input_linear, reset_gate_neighbor_linear, candidate_hidden_linear):

        self._setup_root_prediction()

        padding = create_var_int(torch.zeros(self.hidden_size), False)

        for step_number in range(self.longest_trace):

            current_dfs_steps, tree_indices = self._collect_steps(step_number)
            self.current_batch_list = tree_indices
            self._process_dfs_step_node_prediction(current_dfs_steps, padding)
            self._process_dfs_step_stop_prediction(current_dfs_steps, padding)

            


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

    def forward(self, tree_data, latent_space_tree_vecs):

        
        traces_handler = TracesHandler(tree_data)
        traces_handler.process(self.hidden_size, self.update_gate_linear, self.reset_gate_neighbor_linear, self.reset_gate_input_linear, self.cand)
        
        return
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



