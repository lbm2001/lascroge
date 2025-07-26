import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helper import create_var_int, create_var_float, depth_first_search, GRU, ModTree, TreeNode


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


    def run_dfs(self, stack, x, fa_idx):
        for y in x.neighbors:
            if y.idx == fa_idx: continue
            stack.append( (x,y,1) )
            depth_first_search(stack, y, x.idx)
            stack.append( (y,x,0) )


class MessagePassingHandler(object):

    def __init__(self, tree_batch, max_nb):
        
        self.tree_batch = tree_batch

        dfs_handler = DFSHandler()
        self.traces = dfs_handler.get_traces(tree_batch)

        self.longest_trace = max([len(tr) for tr in self.traces])
        self.batch_size = len(self.tree_batch)
        self.current_batch_list = []

        self.node_prediction_data = NodePredictionData()
        self.stop_prediction_data = StopPredictionData()
        self.hidden_states_directed_edges = {}

        self.max_nb = max_nb
    

    def _setup_root_prediction(self, hidden_size):
        self.node_prediction_data.hidden_states.append(create_var_int(torch.zeros(self.batch_size, hidden_size))) # Initial hidden states for root prediction
        self.node_prediction_data.targets.extend([tree.nodes[0].features for tree in self.tree_batch]) # Root nodes features
        self.node_prediction_data.contexts.append( create_var_int( torch.LongTensor(range(self.batch_size)) ) ) # Creates batch indices
    

    def _collect_steps(self, step_number):
        """Collects all steps at the current DFS step across all trees"""
        current_steps = [] # was prop_list
        batch_list = [] 

        for tree_index, trace in enumerate(self.traces):
            if step_number < len(trace):
                current_steps.append(trace[step_number])
                batch_list.append(tree_index)

        return current_steps, batch_list


    def _extract_neighbor_data(self, current_dfs_steps, padding, hidden_size):
        
        node_prediction_neighbor_hidden_states = [] # was cur_h_nei
        stop_prediction_neighbor_hidden_states = [] # was cur_o_nei
        node_features = [] # was cur_x

        for source_node, target_node, _ in current_dfs_steps:
            

            #Neighbors for message passing (target not included)
            neighbor_hidden_states_wo_target = [self.hidden_states_directed_edges[(neighbor.idx, source_node.idx)] for neighbor in source_node.neighbors if neighbor.idx != target_node.idx]
            padding_length = self.max_nb - len(neighbor_hidden_states_wo_target)
            node_prediction_neighbor_hidden_states.extend(neighbor_hidden_states_wo_target)
            node_prediction_neighbor_hidden_states.extend([padding] * padding_length)

            #Neighbors for stop prediction (all neighbors)
            neighbor_hidden_states = [self.hidden_states_directed_edges[(node_y.idx,source_node.idx)] for node_y in source_node.neighbors]
            padding_length = self.max_nb - len(neighbor_hidden_states)
            stop_prediction_neighbor_hidden_states.extend(neighbor_hidden_states)
            stop_prediction_neighbor_hidden_states.extend([padding] * padding_length)

            node_features.append(source_node.features)
        
        node_prediction_neighbor_hidden_states = torch.stack(node_prediction_neighbor_hidden_states, dim=0).view(-1, self.max_nb, hidden_size)
        stop_prediction_neighbor_hidden_states = torch.stack(stop_prediction_neighbor_hidden_states, dim=0).view(-1, self.max_nb, hidden_size)
        stop_prediction_neighbor_hidden_states = stop_prediction_neighbor_hidden_states.sum(dim=1)
        node_features = create_var_int(torch.FloatTensor(np.array(node_features)))

        return node_prediction_neighbor_hidden_states, stop_prediction_neighbor_hidden_states, node_features


    def _store_node_prediction_data(self, node_prediction_targets, node_prediction_list, new_hidden_states):
          #Hidden states for clique prediction
        if len(node_prediction_list) > 0:
                batch_list = [self.current_batch_list[i] for i in node_prediction_list]
                self.node_prediction_data.contexts.append( create_var_int(torch.LongTensor(batch_list)) )

                cur_pred = create_var_int(torch.LongTensor(node_prediction_list))
                self.node_prediction_data.hidden_states.append( new_hidden_states.index_select(0, cur_pred) )
                self.node_prediction_data.targets.extend( node_prediction_targets )


    def _store_stop_prediction_data(self, stop_prediction_neighbor_hidden_states, stop_target, node_features):
        #Hidden states for stop prediction
        cur_batch = create_var_int(torch.LongTensor(self.current_batch_list))
        stop_hidden = torch.cat([node_features, stop_prediction_neighbor_hidden_states], dim=1)
        self.stop_prediction_data.hidden_states.append( stop_hidden )
        self.stop_prediction_data.contexts.append( cur_batch )
        self.stop_prediction_data.targets.extend( stop_target )


    def _collect_prediction_targets(self, current_dfs_steps, new_hidden_states):
        node_prediction_targets = []
        node_prediction_list = []
        stop_target = []

        for i, step in enumerate(current_dfs_steps):
            source_node, target_node, direction = step
            self.hidden_states_directed_edges[(source_node.idx ,target_node.idx)] = new_hidden_states[i]
            target_node.neighbors.append(source_node)
            if direction == 1:
                node_prediction_targets.append(target_node.features)
                node_prediction_list.append(i) 
            stop_target.append(direction)
        
        return node_prediction_targets, node_prediction_list, stop_target


    def _root_stop(self, padding):
        
        features = []
        neighbors = []

        for tree in self.tree_batch:
            root = tree.nodes[0]
            features.append(root.features)
            current_neighbors = [self.hidden_states_directed_edges[root_neighbor.idx, root.idx] for root_neighbor in root.neighbors]
            pad_len = self.max_nb  - len(current_neighbors)
            neighbors.extend(current_neighbors)
            neighbors.extend([padding] * pad_len)
        
        return features, neighbors


    def message_passing(self, hidden_size, update_gate_linear, reset_gate_input_linear, reset_gate_neighbor_linear, candidate_hidden_linear):

        self._setup_root_prediction(hidden_size)

        padding = create_var_int(torch.zeros(hidden_size), False)

        # Replacement of first big loop
        for step_number in range(self.longest_trace):

            current_dfs_steps, batch_list = self._collect_steps(step_number)
            self.current_batch_list = batch_list
            node_prediction_neighbor_hidden_states, stop_prediction_neighbor_hidden_states, node_features = self._extract_neighbor_data(current_dfs_steps, padding, hidden_size)

            new_hidden_states = GRU(node_features, node_prediction_neighbor_hidden_states, update_gate_linear, reset_gate_input_linear, reset_gate_neighbor_linear, candidate_hidden_linear)
            node_prediction_targets, node_prediction_list, stop_target = self._collect_prediction_targets(current_dfs_steps, new_hidden_states)

            self._store_node_prediction_data(node_prediction_targets, node_prediction_list, new_hidden_states)
            self._store_stop_prediction_data(stop_prediction_neighbor_hidden_states, stop_target, node_features)


        features, neighbors = self._root_stop(padding)

        features = create_var_int(torch.LongTensor(features))
        neighbors = torch.stack(neighbors, dim=0).view(-1, self.max_nb, hidden_size)
        neighbors_sum = neighbors.sum(dim=1)

        stop_hidden =  torch.cat([
            features.unsqueeze(0) 
            if features.dim() == 0 
            else features, neighbors_sum
            ], dim=1)
        
        self.stop_prediction_data.hidden_states.append(stop_hidden)
        self.stop_prediction_data.contexts.append(create_var_int( torch.LongTensor(range(self.batch_size)) ) )
        self.stop_prediction_data.targets.extend([0] * self.batch_size)
    
        #Predict next clique
        self.node_prediction_data.contexts = torch.cat(self.node_prediction_data.contexts, dim=0)
        self.node_prediction_data.hidden_states = torch.cat(self.node_prediction_data.hidden_states, dim=0)


class Decoder(nn.Module):

    def __init__(self, hidden_size, latent_size, max_nb, feature_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.max_nb = max_nb
        self.feature_dim = feature_dim

        #GRU Weights
        self.W_z = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.U_r = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_h = nn.Linear(2 * self.hidden_size, self.hidden_size)

        #Word Prediction Weights 
        self.W = nn.Linear(self.hidden_size + self.latent_size, self.hidden_size)

        #Stop Prediction Weights
        self.U = nn.Linear(self.hidden_size + self.latent_size, self.hidden_size)
        self.U_i = nn.Linear(2 * self.hidden_size, self.hidden_size)

        #Output Weights
        self.W_o = nn.Linear(self.hidden_size, self.feature_dim)
        self.U_o = nn.Linear(self.hidden_size, 1)

        #Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')


    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'features':
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
        
        mp_handler = MessagePassingHandler(tree_data, self.max_nb)
        mp_handler.message_passing(self.hidden_size, self.W_z, self.U_r, self.W_r, self.W_h)
        
        pred_hiddens = mp_handler.node_prediction_data.hidden_states
        pred_targets = mp_handler.node_prediction_data.targets
        pred_contexts = mp_handler.node_prediction_data.contexts

        stop_hiddens = mp_handler.stop_prediction_data.hidden_states
        stop_targets = mp_handler.stop_prediction_data.targets
        stop_contexts = mp_handler.stop_prediction_data.contexts

        pred_scores = self.aggregate(pred_hiddens, pred_contexts, latent_space_tree_vecs, 'features')  # Feature prediction: Use aggregate function to predict node features
        
        # Convert pred_targets to tensor for regression
        pred_targets = create_var_int(torch.FloatTensor(pred_targets)) #was LongTensor before

        pred_loss = self.pred_loss(pred_scores, pred_targets) / len(tree_data) #Loss calculation: Compute regression loss for feature prediction

        # Die unteren 3 Zeilen sind unnötig, da wir pred_acc nicht brauchen
        distance_threshold = 10  # Define a threshold for distance
        distances = torch.norm(pred_scores - pred_targets, dim=1)  # Calculate distances
        pred_acc = torch.mean((distances < distance_threshold).float())  # Calculate accuracy based on distance threshold, The percentage of predicted node features that are "close enough" to the true node features (within a distance threshold).
 
        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu(self.U_i(stop_hiddens) )
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, latent_space_tree_vecs, 'stop') #Stop prediction: Use aggregate function to predict stop decisions
        stop_scores = stop_scores.squeeze(-1)  # Simplified placeholder
        stop_targets = create_var_float(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss(stop_scores, stop_targets) / len(tree_data)  
        # Die unteren 3 Zeilen brauchen wir nicht, stop_acc ist unnötig
        stops = torch.ge(stop_scores, 0).float() #checks if each score is ≥ 0; If score ≥ 0 → decision is 1 (continue expanding) , If score < 0 → decision is 0 (stop expanding)
        stop_acc = torch.eq(stops, stop_targets).float() #compares each prediction with the correct answer; .float() converts to 1.0 (correct) or 0.0 (incorrect)
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement() #The percentage of nodes where the model correctly predicted whether to stop or continue expanding that branch.

        return pred_loss, stop_loss, pred_acc.item(), stop_acc.item()


    def decode():
        pass


