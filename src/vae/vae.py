import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.helper import create_var_float, create_var_int, dfs, GRU, GraphGRU, index_select_ND
from vae.mod_tree import TreeNode


class Encoder(nn.Module):

    def __init__(self, hidden_size, latent_size, depth, encoding_method, feature_dim):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.depth = depth
        self.encoding_method = encoding_method
        self.feature_dim = feature_dim

        self.gru = GraphGRU(self.hidden_size, self.hidden_size, self.depth)

        self.input_to_hidden = nn.Linear(feature_dim, hidden_size )  # Linear layer to project input features to hidden size
        self.outputNN = nn.Sequential(nn.Linear(2 * hidden_size , hidden_size ), nn.ReLU())

        self.W_mean = nn.Linear(self.hidden_size , self.latent_size)
        self.W_var = nn.Linear(self.hidden_size , self.latent_size)


    def rsample(self, z_vecs): #W_mean and W_var are nn.Linear layers
        batch_size = z_vecs.size(0)
        z_mean = self.W_mean(z_vecs) # Apply linear Layer to compute mean of latent Gaussian distribution, z_mean shape is [batch_size, LATENT_SIZE]
        #Computes log variance, then makes it negative to ensure \sigma^2 <= 1 (log(\sigma^2) <= 0)
        z_log_var = -torch.abs(self.W_var(z_vecs)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size # Dividing by batch_size gives average over batch
        epsilon = create_var_float(torch.randn_like(z_mean)) # epsilon ~ N(0,I)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon # z_vecs = \mu + \sigma * \epsilon
        return z_vecs, kl_loss


    def encode(self, jtenc_holder):
        fnode, fmess, node_graph, mess_graph, scope, leafs = jtenc_holder

        # vae_train.py -> forward() #ÄÄÄ jtnn_enc.py -> forward() oder vae_train.py -> encode()? 
        fnode = create_var_float(fnode)
        fmess = create_var_int(fmess)
        node_graph = create_var_int(node_graph)
        mess_graph = create_var_int(mess_graph)
        messages = create_var_float(torch.zeros(mess_graph.size(0), self.hidden_size))

        fnode = self.input_to_hidden(fnode) # Here we skip the embedding
        fmess = index_select_ND(fnode, 0, fmess)
        messages = self.gru.forward(messages, fmess, mess_graph)

        mess_nei = index_select_ND(messages, 0, node_graph)
        node_vecs = torch.cat([fnode, mess_nei.sum(dim=1)], dim=1)
        node_vecs = self.outputNN(node_vecs)
        
        max_len = max([x for _,x in scope])

        #ÄÄÄ Ab hier wird es deutlich anders bzw. wir haben Encoding "root" weggelassen
        batch_vecs = []
        for leaf in leafs: # Eine Liste von Listen, wobei jede innere Liste Indizes von Knoten enthält, die zu einem „Blatt-Cluster“ gehören. leafs = [[2, 3], [5], [7, 8, 9]] „dieser Teilbaum hat die Blätter (2,3)“, „dieser nur (5)“, etc.
            cur_vecs = torch.zeros_like(node_vecs[0]) #node_vecs is a tensor of shape [n_nodes, HIDDEN_SIZE]??? Eine Liste (oder Tensor) von Feature-Vektoren pro Knoten.
            for node_idx in leaf:
                cur_vecs += node_vecs[node_idx] #Für jeden „Blatt-Cluster“ in leafs: starte mit einem Nullvektor cur_vecs; summiere die Vektoren aller Knoten in diesem Blatt-Cluster auf
            if self.encoding_method == "average":
                cur_vecs /= len(leaf)
            elif self.encoding_method != "sum":
                exit(f"Encoding method is not in the list")
            batch_vecs.append(cur_vecs)

        tree_vecs = torch.stack(batch_vecs, dim=0)
        return tree_vecs, messages #tree_vecs ist output vom Encoder


class Decoder(nn.Module):

    def __init__(self, hidden_size, max_nb, latent_size, feature_dim, joint_dim, body_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_nb = max_nb
        self.latent_size = latent_size
        self.feature_dim = feature_dim
        self.joint_dim = joint_dim
        self.body_dim = body_dim

        #Loss functions
        self.pred_loss_nn = nn.MSELoss(reduction='sum')
        self.stop_loss_nn = nn.BCEWithLogitsLoss(reduction='sum')


        self.W_z = nn.Linear(2 * self.hidden_size , self.hidden_size )
        self.U_r = nn.Linear(self.hidden_size , self.hidden_size , bias=False)
        self.W_r = nn.Linear(self.hidden_size , self.hidden_size )
        self.W_h = nn.Linear(2 * self.hidden_size , self.hidden_size )

        self.U_i = nn.Linear(2 * self.hidden_size , self.hidden_size )

        self.W = nn.Linear(self.hidden_size  + self.latent_size, self.hidden_size )
        self.U = nn.Linear(self.hidden_size  + self.latent_size, self.hidden_size )
        #self.W_o = nn.Linear(self.hidden_size , self.feature_dim)  # Output layer for clique prediction
        self.W_o_categorical = nn.Linear(self.hidden_size, 1)  # For body/joint classification (logits)
        self.W_o_continuous = nn.Linear(self.hidden_size, self.feature_dim - 1)  # For remaining features 1508
        self.U_o = nn.Linear(self.hidden_size , 1)  # Output layer for stop prediction

        self.features_to_dim = nn.Linear(self.feature_dim, self.hidden_size )  # Linear layer to project features to hidden size


    def aggregate(self, hiddens, contexts, x_tree_vecs, mode):
        if mode == 'features_categorical': # Renamed from 'word'
            V, V_o = self.W, self.W_o_categorical
        elif mode == 'features_continuous':
            V, V_o = self.W, self.W_o_continuous
        elif mode == 'stop':
            V, V_o = self.U, self.U_o
        else:
            raise ValueError('aggregate mode is wrong')

        tree_contexts = x_tree_vecs.index_select(0, contexts)
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        output_vec = F.relu( V(input_vec) )
        return V_o(output_vec)


    def forward(self, mol_batch, x_tree_vecs):
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
        pred_hiddens.append(create_var_int(torch.zeros(batch_size,self.hidden_size))) # Initial hidden states are zeros (no previous context)
        pred_targets.extend([mol_tree.nodes[0].features for mol_tree in mol_batch]) #Actual features of root nodes (aktuell noch features vom ersten Knoten)
        pred_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) ) # Maps each prediction to its corresponding tree in the batch

        max_iter = max([len(tr) for tr in traces]) # Maximum number of steps needed (longest trace in batch)
        padding = create_var_int(torch.zeros(self.hidden_size), False) # Zero vector used for padding when nodes have fewer neighbors
        h = {} #Dictionary to store hidden states between node pairs: h[(from_node, to_node)]

        # Create a batch of input
        # All the neighbors are set to [] initially
        # Max_nb: max neighbor
        for t in range(max_iter):
            prop_list = [] #prop_list is the traces flattened (s_11, s_12 , s_21 , s_22) (11 - erster knoten erster baum)
            batch_list = [] # Tracks which tree each active node/step belongs to
            #Collect active nodes at this step
            for i,plist in enumerate(traces):
                if t < len(plist):
                    prop_list.append(plist[t])
                    batch_list.append(i) # Keep track of which tree each active node belongs to

            # cur_x current node features
            cur_x = []
            # cur_h_nei Neighbor hidden states for message passing
            # cur_o_nei Neighbor hidden states for stop predicition
            cur_h_nei,cur_o_nei = [],[]

            #Process each active node
            for node_x, real_y, _ in prop_list:
                #Neighbors for message passing (target not included), der Knoten zu dem wir gehen (real_y) wird ausgeschlossen
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != real_y.idx] #For each active node, collect neighbor hidden states
                pad_len = self.max_nb - len(cur_nei) # Pad to fixed size self.max_nb (maximum neighbors) for batch processing
                cur_h_nei.extend(cur_nei)
                cur_h_nei.extend([padding] * pad_len)

                #Neighbors for stop prediction (all neighbors)
                cur_nei = [h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors] #Stop prediction: Include ALL neighbors (different from message passing)
                pad_len = self.max_nb - len(cur_nei) 
                cur_o_nei.extend(cur_nei)
                cur_o_nei.extend([padding] * pad_len)

                #Current node features
                cur_x.append(node_x.features) #Node features: Collect current node's features

            # Convert features to tensor (no embedding needed)
            #cur_x = torch.tensor(cur_x, dtype=torch.float32)
            #if torch.cuda.is_available():
            #    cur_x = cur_x.cuda()
            #Clique embedding
            cur_x = create_var_float(torch.FloatTensor(np.array(cur_x, dtype=np.float32)))
            cur_x = self.features_to_dim(cur_x)  # Skip embedding for now 
            
            #Message passing
            cur_h_nei = torch.stack(cur_h_nei, dim=0).view(-1,self.max_nb,self.hidden_size)
            new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h) #GRU message passing: Update hidden states using GRU with neighbor information
  
            #Node Aggregate
            cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,self.max_nb,self.hidden_size) # Was soll self.max_nb sein !!!
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
            stop_hidden = torch.cat([cur_x, cur_o], dim=1) #cur_x current node features, cur_o aggregated neighbor information
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
            pad_len = self.max_nb - len(cur_nei) # Was soll self.max_nb sein !!!
            cur_o_nei.extend(cur_nei)
            cur_o_nei.extend([padding] * pad_len)

        
        # Convert features to tensor (no embedding needed)
        #cur_x = torch.tensor(cur_x, dtype=torch.float32)
        #if torch.cuda.is_available():
        #    cur_x = cur_x.cuda()
        cur_x = create_var_float(torch.FloatTensor(np.array(cur_x, dtype=np.float32)))
        cur_x = self.features_to_dim(cur_x) # Embedding skipped for now

        cur_o_nei = torch.stack(cur_o_nei, dim=0).view(-1,self.max_nb,self.hidden_size) # Was soll self.max_nb sein !!!
        cur_o = cur_o_nei.sum(dim=1)

        stop_hidden = torch.cat([cur_x.unsqueeze(0) if cur_x.dim() == 0 else cur_x, cur_o], dim=1) #ÄÄÄ Wieder unsqueeze
        stop_hiddens.append( stop_hidden )
        stop_contexts.append( create_var_int( torch.LongTensor(range(batch_size)) ) ) #ÄÄÄ Wieder create_var_int
        stop_targets.extend( [0] * len(mol_batch) )

        #Predict next clique
        pred_contexts = torch.cat(pred_contexts, dim=0)
        pred_hiddens = torch.cat(pred_hiddens, dim=0)
        #pred_scores = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'features')  # Feature prediction: Use aggregate function to predict node features
        pred_scores_categorical = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'features_categorical')  # Feature prediction: Use aggregate function to predict node features
        pred_scores_continuous = self.aggregate(pred_hiddens, pred_contexts, x_tree_vecs, 'features_continuous') # 1508
        
        #print(f"Prediction targets: {pred_targets}")    
        #print(f"Stop targets in training: {stop_targets}")

        # Convert pred_targets to tensor for regression
        pred_targets = torch.tensor(np.array(pred_targets, dtype=np.float32), dtype=torch.float32)
        pred_targets_tensor = create_var_int(torch.FloatTensor(np.array(pred_targets))) #was LongTensor before
        pred_targets_categorical = pred_targets_tensor[:, 0:1] # First Column
        pred_targets_continuous = pred_targets_tensor[:, 1:]  # Remaining columns

        categorical_loss = F.binary_cross_entropy_with_logits(pred_scores_categorical, pred_targets_categorical, reduction='sum') / len(mol_batch) # Categorical loss for feature prediction   1508
        continuous_loss = self.pred_loss_nn(pred_scores_continuous, pred_targets_continuous) / len(mol_batch)  # Regression loss for feature prediction 1508
        pred_loss = categorical_loss + continuous_loss  # Combine losses for feature prediction

        #pred_loss = self.pred_loss_nn(pred_scores, pred_targets) / len(mol_batch) #Loss calculation: Compute regression loss for feature prediction

        # Die unteren 3 Zeilen sind unnötig, da wir pred_acc nicht brauchen
        #distnace_threshold = 10  # Define a threshold for distance
        #distances = torch.norm(pred_scores - pred_targets, dim=1)  # Calculate distances
        #pred_acc = torch.mean((distances < distnace_threshold).float())  # Calculate accuracy based on distance threshold, The percentage of predicted node features that are "close enough" to the true node features (within a distance threshold).

        #Predict stop
        stop_contexts = torch.cat(stop_contexts, dim=0)
        stop_hiddens = torch.cat(stop_hiddens, dim=0)
        stop_hiddens = F.relu(self.U_i(stop_hiddens) )
        stop_scores = self.aggregate(stop_hiddens, stop_contexts, x_tree_vecs, 'stop') #Stop prediction: Use aggregate function to predict stop decisions
        stop_scores = stop_scores.squeeze(-1)  # Simplified placeholder
        stop_targets = create_var_float(torch.Tensor(stop_targets))
        
        stop_loss = self.stop_loss_nn(stop_scores, stop_targets) / len(mol_batch)  
        # Die unteren 3 Zeilen brauchen wir nicht, stop_acc ist unnötig
        stops = torch.ge(stop_scores, 0).float() #checks if each score is ≥ 0; If score ≥ 0 → decision is 1 (continue expanding) , If score < 0 → decision is 0 (stop expanding)
        stop_acc = torch.eq(stops, stop_targets).float() #compares each prediction with the correct answer; .float() converts to 1.0 (correct) or 0.0 (incorrect)
        stop_acc = torch.sum(stop_acc) / stop_targets.nelement() #The percentage of nodes where the model correctly predicted whether to stop or continue expanding that branch.

        return pred_loss, stop_loss, stop_acc.item()


    def decode(self, x_tree_vecs, prob_decode, max_decode_len):
            assert x_tree_vecs.size(0) == 1 

            stack = []
            init_hiddens = create_var_int( torch.zeros(1, self.hidden_size) )
            zero_pad = create_var_int(torch.zeros(1,1,self.hidden_size))
            contexts = create_var_int( torch.LongTensor(1).zero_() ) #!!! Macht der Zero Vector hier sinn?

            #Root Prediction
            #root_features = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'features') 1508
            root_categorical = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'features_categorical')
            root_continuous = self.aggregate(init_hiddens, contexts, x_tree_vecs, 'features_continuous')
            root_categorical_binary = (torch.sigmoid(root_categorical) > 0.5).float() #1508 Wir nutzen nicht direkt Sigmoid im Layer, da wir  F.binary_cross_entropy_with_logits() nutzen -> more stable
            #1508 Problem mit diesem Ansatz: Wir dürfen Sigmoid bei der Inferenz/Decode nicht vergessen
            root_features = torch.cat([root_categorical_binary.view(-1), root_continuous.view(-1)], dim=-1)
            #_,root_wid = torch.max(root_score, dim=1) apparently this is not needed, as it is for classification
            #root_wid = root_wid.item() not needed as we have attribute features

            root = TreeNode(root_features) #root = TreeNode(root_features.squeeze().detach()) ???
            #root.wid = root_wid
            root.idx = 0
            stack.append( (root, None) )

            all_nodes = [root]
            h = {}
            for step in range(max_decode_len):
                node_x, _ = stack[-1]
                cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors ]
                if len(cur_h_nei) > 0:
                    cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1, self.hidden_size)
                else:
                    cur_h_nei = zero_pad

                cur_x = create_var_float(node_x.features.detach().clone().float().unsqueeze(0))
                cur_x = self.features_to_dim(cur_x)
                cur_x = cur_x.squeeze(1)
                
                #Predict stop
                cur_h = cur_h_nei.sum(dim=1)
                # Debug: Print shapes to identify the exact issue

                stop_hiddens = torch.cat([cur_x,cur_h], dim=1)
                stop_hiddens = F.relu(self.U_i(stop_hiddens) )
            
                stop_score = self.aggregate(stop_hiddens, contexts, x_tree_vecs, 'stop')
                
                if prob_decode:
                    backtrack = (torch.bernoulli( torch.sigmoid(stop_score) ).item() == 0)
                else:
                    backtrack = (stop_score.item() < 0)
                    # print(f'step = {step}, backtrack = {backtrack}, stopscore = {stop_score}')
                
                if not backtrack: #Forward: Predict next clique
                    
                    new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)

                    #pred_features = self.aggregate(new_h, contexts, x_tree_vecs, 'features')

                    # For regression, we directly use the predicted features
                    # No need for vocabulary sampling or sorting
                    #predicted_features = pred_features.squeeze().detach()  # Get the predicted features for the next node
                    pred_categorical = self.aggregate(new_h, contexts, x_tree_vecs, 'features_categorical')
                    pred_continuous = self.aggregate(new_h, contexts, x_tree_vecs, 'features_continuous')
                    pred_categorical_binary = (torch.sigmoid(pred_categorical) > 0.5).float()
                    predicted_features = torch.cat([pred_categorical_binary.view(-1), pred_continuous.view(-1)], dim=-1).squeeze().detach()

                    node_y = TreeNode(predicted_features)
                    node_y.idx = len(all_nodes)
                    node_y.neighbors.append(node_x)
                    #node_x.neighbors.append(node_y)  #ÄÄÄ später hinzugefügt
                    h[(node_x.idx,node_y.idx)] = new_h[0]
                    stack.append( (node_y, None) )
                    all_nodes.append(node_y)
                    

                if backtrack: #Backtrack, use if instead of else
                    
                    if len(stack) == 1:
                        
                        break #At root, terminate

                    node_fa,_ = stack[-2]
                    cur_h_nei = [ h[(node_y.idx,node_x.idx)] for node_y in node_x.neighbors if node_y.idx != node_fa.idx ]
                    if len(cur_h_nei) > 0:
                        cur_h_nei = torch.stack(cur_h_nei, dim=0).view(1,-1,self.hidden_size)
                    else:
                        cur_h_nei = zero_pad

                    new_h = GRU(cur_x, cur_h_nei, self.W_z, self.W_r, self.U_r, self.W_h)
                    h[(node_x.idx,node_fa.idx)] = new_h[0]
                    node_fa.neighbors.append(node_x)
                    stack.pop()
                    

            return root, all_nodes


class VAE(nn.Module):
    def __init__(self, hidden_size, latent_size, feature_dim, max_decode_len, depth, encoding_method, max_nb, joint_dim, body_dim):
        super(VAE, self).__init__()

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.feature_dim = feature_dim
        self.max_decode_len = max_decode_len
        self.depth = depth
        self.encoding_method = encoding_method
        self.max_nb = max_nb
        self.joint_dim = joint_dim
        self.body_dim = body_dim

        self.encoder = Encoder(self.hidden_size, self.latent_size, self.depth, self.encoding_method, feature_dim=self.feature_dim)
        self.decoder = Decoder(self.hidden_size, self.max_nb, self.latent_size, self.feature_dim, self.joint_dim, self.body_dim)


    def forward(self, batch, beta, alpha, gamma):
        tree_batch, jtenc_holder = batch
        res = self.encoder.encode(jtenc_holder)
        tree_vecs = res[0]
        messages = res[1]

        z_tree_vecs, kl_div = self.encoder.rsample(z_vecs=tree_vecs)

        pred_loss, stop_loss, stop_acc = self.decoder.forward(tree_batch, z_tree_vecs)
        total_loss = pred_loss + stop_loss + beta * kl_div

        return total_loss, kl_div, stop_acc, pred_loss, stop_loss
    

    def decode(self, z_tree_vecs, prob_decode):
        
        root, all_nodes = self.decoder.decode(z_tree_vecs, prob_decode=prob_decode, max_decode_len=self.max_decode_len)
        
        return root, all_nodes
