import torch  # PyTorch tensor library
import torch.nn as nn  
import numpy as np  
import yaml
import wandb  

from vae.vae import VAE
from vae.helper import tensorize, tree_to_adjacency

torch.manual_seed(42)



training_config_path = r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\src\train_config.yml"

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
#adj_matrices = np.load(input_data_paths["adj_matrices"], allow_pickle=True).astype(np.int64)
#features = np.load(input_data_paths["features"], allow_pickle=True).astype(np.float32)
adj_matrices = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\adj.npy", allow_pickle=True)
features = np.load(r"C:\Users\nurha\OneDrive\Desktop\UNI\lascroge\data\robot_graphs\feat.npy", allow_pickle=True)
for i in range(len(adj_matrices)):
    min_size = min(adj_matrices[i].shape[0], features[i].shape[0])
    adj_matrices[i] = adj_matrices[i][:min_size, :min_size]
    features[i] = features[i][:min_size, :]
training_data_size = len(adj_matrices)

np.set_printoptions(threshold=np.inf, linewidth=200)
# ========= Model Save Path =========
model_path = config["model_save_path"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(num_epochs, beta, alpha, gamma, model_save_path):

    wandb.init(
        project="glso-vae",
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        #print(f"{epoch} of {num_epochs}")

        if epoch % 100 == 0 and epoch > 0:
            scheduler.step()

        batch = tensorize(features, adj_matrices)
        
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

    batch = tensorize(features, adj_matrices) 
    _, jtenc_holder = batch
    res = model.encoder.encode(jtenc_holder)
    tree_vecs = res[0]

    z_tree_vecs, _ = model.encoder.rsample(z_vecs=tree_vecs)
    
    z_single = z_tree_vecs[0:1]  # Take the first tree vector for testing
    root, all_nodes = model.decode(z_single, prob_decode=False)
    tree = tree_to_adjacency(root)
    """
    print("Decoded tree structure:")
    print("Number of nodes in decoded tree:", len(all_nodes))
    for i, node in enumerate(all_nodes):
        print(f"Node {i}: {node.features}")
    print(tree)
    """
    
    for i in range(training_data_size):
        z_single = z_tree_vecs[i:i+1]

        root, all_nodes = model.decode(z_single, prob_decode=False)
        tree = tree_to_adjacency(root)

        print("Decoded tree structure:")
        print(f"Decoded tree root: {root.features}")
        print(f"Number of nodes in decoded tree: {len(all_nodes)}")
        #for i,node in enumerate(all_nodes):
        #    print(f"Node {i}: {node.features}")
        #print(tree)
        print("\n")
        

import os

if __name__ == "__main__":       
      #train_loop(num_epochs=NUM_EPOCHS, beta=BETA, alpha=ALPHA, gamma=GAMMA, model_save_path=model_path)
      test_decoder(model_load_path=model_path)
      #print(adj_matrices[0].shape)
      #print(features[0])
      #print(features[0].shape)

