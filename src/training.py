import torch  # PyTorch tensor library
import torch.nn as nn  
import numpy as np  
import yaml
import wandb  

from vae.vae import VAE
from vae.helper import tensorize, tree_to_adjacency

torch.manual_seed(42)



training_config_path = "/Users/lukasmueller/github/lascroge/src/train_config.yml"

with open(training_config_path, "r") as file:
    config = yaml.safe_load(file)

params = config["parameters"]
training_params = config["training_parameters"]
input_data_paths = config["input_data_paths"]

# ========== Parameters ==========
HIDDEN_SIZE = params['hidden_size']
LATENT_SIZE = params['latent_size']
DEPTH = params["depth"]
ENCODING_METHOD = params["encoding_method"]
FEATURE_DIM = params["feature_dim"] # Should be the number of features per node, e.g. 3 for [1, 5, 6]
MAX_NB = params["max_nb"]
MAX_DECODE_LEN = params["max_decode_len"]


# ========== Training Parameters =========
NUM_EPOCHS = training_params["num_epochs"]
BETA = training_params["beta"]
ALPHA = training_params["alpha"]
GAMMA = training_params["gamma"]


# ========== Input Data ==========
adj_matrices = np.load(input_data_paths["adj_matrices"], allow_pickle=True)
features = np.load(input_data_paths["features"], allow_pickle=True)
#adj_matrices = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\adj.npy", allow_pickle=True)
#features = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\feat.npy", allow_pickle=True)
training_data_size = len(adj_matrices)

np.set_printoptions(threshold=np.inf, linewidth=200)
# ========= Model Save Path =========
model_path = config["model_save_path"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_normalization_params(features):
    """Compute separate normalization parameters for different node types and feature dimensions"""
    # Separate features by node type based on first feature (0=body/geom, 1=joint)
    body_features = []  # includes geom features (type=0)
    joint_features = []  # type=1
    
    for graph_feats in features:
        for node_feat in graph_feats:
            if node_feat[0] == 0:  # body or geom node
                body_features.append(node_feat[1:])  # exclude type indicator
            elif node_feat[0] == 1:  # joint node  
                joint_features.append(node_feat[1:])  # exclude type indicator
    
    body_features = np.array(body_features) 
    joint_features = np.array(joint_features)     
    
    body_means = np.mean(body_features, axis=0)
    body_stds = np.std(body_features, axis=0)
    # Handle features with zero std (constant features)
    body_stds = np.where(body_stds < 1e-8, 1.0, body_stds)
    
    joint_means = np.mean(joint_features, axis=0)
    joint_stds = np.std(joint_features, axis=0)
    # Handle features with zero std (constant features)
    joint_stds = np.where(joint_stds < 1e-8, 1.0, joint_stds)
    
    return body_means, body_stds, joint_means, joint_stds

def normalize_features(features, body_means, body_stds, joint_means, joint_stds):
    """Apply type-specific Z-score normalization to features"""
    normalized_features = []
    
    for graph_feats in features:
        normalized_graph = np.zeros_like(graph_feats)
        
        for i, node_feat in enumerate(graph_feats):
            node_type = node_feat[0]
            normalized_graph[i, 0] = node_type  # Keep type indicator unchanged
            
            if node_type == 0:  # body/geom node
                normalized_graph[i, 1:] = (node_feat[1:] - body_means) / body_stds
            elif node_type == 1:  # joint node
                normalized_graph[i, 1:] = (node_feat[1:] - joint_means) / joint_stds
        
        normalized_features.append(normalized_graph)
    
    return normalized_features

def denormalize_features(normalized_features, norm_params):
    """Convert normalized features back to original scale"""
    if isinstance(normalized_features, torch.Tensor):
        normalized_features = normalized_features.detach().cpu().numpy()
    
    denormalized = np.zeros_like(normalized_features)
    
    if normalized_features.ndim == 1:  # Single node
        node_type = int(normalized_features[0])
        denormalized[0] = node_type
        
        if node_type == 0 and 'body_means' in norm_params:
            denormalized[1:] = (normalized_features[1:] * norm_params['body_stds']) + norm_params['body_means']
        elif node_type == 1 and 'joint_means' in norm_params:
            denormalized[1:] = (normalized_features[1:] * norm_params['joint_stds']) + norm_params['joint_means']
        else:
            denormalized[1:] = normalized_features[1:]
    else:  # Multiple nodes
        for i in range(normalized_features.shape[0]):
            node_type = int(normalized_features[i, 0])
            denormalized[i, 0] = node_type
            
            if node_type == 0 and 'body_means' in norm_params:
                denormalized[i, 1:] = (normalized_features[i, 1:] * norm_params['body_stds']) + norm_params['body_means']
            elif node_type == 1 and 'joint_means' in norm_params:
                denormalized[i, 1:] = (normalized_features[i, 1:] * norm_params['joint_stds']) + norm_params['joint_means']
            else:
                denormalized[i, 1:] = normalized_features[i, 1:]
    
    return denormalized

def train_loop(num_epochs, beta, alpha, gamma, model_save_path):

    print("Computing normalization parameters for different node types...")
    body_means, body_stds, joint_means, joint_stds = compute_normalization_params(features)
    print("Normalizing features by node type...")
    normalized_features = normalize_features(features, body_means, body_stds, joint_means, joint_stds)
    normalized_features = features


    norm_params = {
    'body_means': body_means,
    'body_stds': body_stds, 
    'joint_means': joint_means,
    'joint_stds': joint_stds
    }
    norm_path = model_save_path.replace('.pth', '_norm_params.npy')
    np.save(norm_path, norm_params)
    print(f"Normalization parameters saved to: {norm_path}")

    wandb.init(
        project="glso-vae",
        name="Synthetic Dataset Run",
        config={
            "learning_rate": 0.001,
            "beta": beta,
            "alpha": alpha,
            "gamma": gamma,
            "epochs": num_epochs,
            "hidden_size": HIDDEN_SIZE,
            "latent_size": LATENT_SIZE,
            "depth": DEPTH,
            #"batch_size": BATCH_SIZE,
        }
    )

    model = VAE(hidden_size=HIDDEN_SIZE,   
                latent_size=LATENT_SIZE, 
                feature_dim=FEATURE_DIM, 
                max_decode_len=MAX_DECODE_LEN, 
                depth=DEPTH, 
                encoding_method=ENCODING_METHOD,
                max_nb=MAX_NB).to(device)
    
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        #print(f"{epoch} of {num_epochs}")

        if epoch % 100 == 0 and epoch > 0:
            scheduler.step()

        batch = tensorize(normalized_features, adj_matrices)
        
        model.zero_grad()
        loss, kl_div, tacc, pred_loss, stop_loss = model(batch, beta, alpha, gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 50.0)  # Gradient clipping
        optimizer.step()

        wandb.log({
                "epoch": epoch,
                "loss": loss.item(),
                "pred_loss": pred_loss.item(),
                "stop_loss": stop_loss.item(),
                "kl_div": kl_div.item(),
                "beta": beta,
                "learning_rate": scheduler.get_last_lr()[0]
                 })
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss={loss.item(): .4f}, Stop Acc={tacc}, PredLoss={pred_loss.item(): .4f}, StopLoss={stop_loss.item(): .4f}, KL Divergence={kl_div.item(): .4f}")
        
    torch.save(model.state_dict(), model_save_path)
    wandb.save('trained_model.pth')
    print("Model saved after epoch", epoch)
    wandb.finish()


def test_decoder(model_load_path):

    model = VAE(hidden_size=HIDDEN_SIZE, 
                latent_size=LATENT_SIZE, 
                feature_dim=FEATURE_DIM, 
                max_decode_len=MAX_DECODE_LEN, 
                depth=DEPTH, 
                encoding_method=ENCODING_METHOD, 
                max_nb=MAX_NB).to(device)
    
    model.load_state_dict(torch.load(model_load_path))
    #model.eval()

    norm_path = model_load_path.replace('.pth', '_norm_params.npy')
    norm_params = None
    if os.path.exists(norm_path):
        norm_params = np.load(norm_path, allow_pickle=True).item()
        print("Loaded normalization parameters:")
        # Normalize features for inference (using training parameters)
        body_means = norm_params['body_means']
        body_stds = norm_params['body_stds']
        joint_means = norm_params['joint_means']
        joint_stds = norm_params['joint_stds']
        normalized_features = normalize_features(features, body_means, body_stds, joint_means, joint_stds)
    else:
        print("Warning: No normalization parameters found. Using raw features.")
        normalized_features = features

    batch = tensorize(normalized_features, adj_matrices) 
    _, jtenc_holder = batch

    # with torch.no_grad():
    res = model.encoder.encode(jtenc_holder)
    tree_vecs = res[0]

    z_tree_vecs, _ = model.encoder.rsample(z_vecs=tree_vecs)
    
    z_single = z_tree_vecs[0:1]  # Take the first element for testing
    root, all_nodes = model.decode(z_single, prob_decode=False)

    denormalized_root_features = denormalize_features(root.features, norm_params)
    print("Decoded tree structure (denormalized root features):")
    print(denormalized_root_features)
    print("original root features:")
    print(features[0][0])

    #print("Decoded tree structure:")
    #print(f"Decoded tree root: {root.features}")
    print(f"Number of nodes in decoded tree: {len(all_nodes)}")
    #for i, node in enumerate(all_nodes):
    #    print(f"Node {i}: {node.features}")
    """
    for i in range(training_data_size):
        z_single = z_tree_vecs[i:i+1]

        root, all_nodes = model.decode(z_single, prob_decode=False)
        tree = tree_to_adjacency(root)

        print("Decoded tree structure:")
        print(f"Decoded tree root: {root.features}")
        print(f"Number of nodes in decoded tree: {len(all_nodes)}")
        for i,node in enumerate(all_nodes):
            print(f"Node {i}: {node.features}")
        print(tree)
        print("\n")
        """

import os

if __name__ == "__main__":       
      train_loop(num_epochs=NUM_EPOCHS, beta=BETA, alpha=ALPHA, gamma=GAMMA, model_save_path=model_path)
      #test_decoder(model_load_path=model_path)
      #print(adj_matrices[0].shape)
      #print(features[0].shape)
      #print(adj_matrices[1].shape)
      #print(features[1].shape)
