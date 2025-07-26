
import torch
import numpy as np
from mod_tree import ModTree


class TreeBatchProcessor(object):

    def __init__(self, adj_matrices, node_features):
        self.batch = self._build_tree_batch(adj_matrices, node_features)
        self._set_node_indices(self.batch)


    def _build_tree_batch(self, adj_matrices, node_features):
        batch = []
        for idx in range(len(node_features)):
            tree = ModTree(node_features[idx], adj_matrices[idx])
            batch.append(tree)

        return batch


    def _set_node_indices(self, batch):
        total = 0
        for tree in batch:
            for node in tree.nodes:
                node.idx = total
                total += 1


    def _get_leafs(self, tree):
        leafs = []
        for node in tree.nodes:
            if len(node.neighbors) == 1:
                leafs.append(node.idx)
        
        return leafs


    def _build_node_batch_and_scope(self, batch):
        
        node_batch = []
        scope = []
        leafs = []

        for tree in batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

            current_tree_leafs = self._get_leafs(tree)
            leafs.append(current_tree_leafs)
        
        return node_batch, scope, leafs


    def _build_messages(self, node_batch):
        message_pairs = [None]
        message_index_map = {}
        node_features = []

        for node in node_batch: 
            node_features.append(node.features) 
            for neighbor in node.neighbors:
                message_index_map[(node.idx, neighbor.idx)] = len(message_pairs) 
                message_pairs.append( (node, neighbor) )

        return message_pairs, message_index_map, node_features


    def _build_graph_structure(self, node_batch, message_pairs, message_index_map):
        incoming_message_indices = [[] for _ in range(len(node_batch))]
        message_dependencies = [[] for _ in range(len(message_pairs))]
        message_source_nodes = [0] * len(message_pairs)

        for source_node, target_node in message_pairs[1:]:
            current_message_idx = message_index_map[(source_node.idx, target_node.idx)]
            message_source_nodes[current_message_idx] = source_node.idx
            incoming_message_indices[target_node.idx].append(current_message_idx)

            for next_hop_node in target_node.neighbors:
                if next_hop_node.idx == source_node.idx: continue

                outgoing_message_idx = message_index_map[(target_node.idx, next_hop_node.idx)]
                message_dependencies[outgoing_message_idx].append(current_message_idx)

        return incoming_message_indices, message_dependencies, message_source_nodes


    def _pad_and_tensorize(self, incoming_message_indices, message_dependencies, message_source_nodes, node_features):
        # Pad incoming_message_indices
        max_len = max([len(indices) for indices in incoming_message_indices] + [1])
        for indices in incoming_message_indices:
            pad_len = max_len - len(indices)
            indices.extend([0] * pad_len)

        # Pad message_dependencies  
        max_len = max([len(deps) for deps in message_dependencies] + [1])
        for deps in message_dependencies:
            pad_len = max_len - len(deps)
            deps.extend([0] * pad_len)

        # Convert to tensors
        incoming_message_indices = torch.LongTensor(incoming_message_indices)
        message_dependencies = torch.LongTensor(message_dependencies)
        message_source_nodes = torch.LongTensor(message_source_nodes)
        node_features = torch.LongTensor(np.array(node_features))
        
        return node_features, message_source_nodes, incoming_message_indices, message_dependencies


    def get_batch(self):
        return self.batch


    def prepare_encoding(self): 
        node_batch, scope, leafs = self._build_node_batch_and_scope(self.batch) 
        message_pairs, message_index_map, node_features = self._build_messages(node_batch)
        incoming_message_indices, message_dependencies, message_source_nodes = self._build_graph_structure(node_batch, message_pairs, message_index_map)
        
        node_features_tensor, message_source_tensor, incoming_messages_tensor, message_deps_tensor = self._pad_and_tensorize(
            incoming_message_indices, message_dependencies, message_source_nodes, node_features)
        
        return node_features_tensor, message_source_tensor, incoming_messages_tensor, message_deps_tensor, scope, leafs
