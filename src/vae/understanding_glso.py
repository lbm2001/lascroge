import torch  # PyTorch tensor library
import torch.nn as nn  
import numpy as np  
import wandb  

from vae import VAE
from mock_data import cur_conn, cur_attr
from helper import tensorize, tree_to_adjacency

torch.manual_seed(42)


# ========== Parameters ==========
HIDDEN_SIZE = 300
LATENT_SIZE = 16
DEPTHT = 3
ENCODING_METHOD = "average"
MAX_NB = 4
FEATURE_DIM = 3 # Should be the number of features per node, e.g. 3 for [1, 5, 6]
MAX_NB = 4  # Maximum number of neighbors per node, can be adjusted based on the dataset
MAX_DECODE_LEN = 100

# ========== Training Parameters ======== 
num_epochs = 5000
beta = 0.001
alpha = 1.0
gamma = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop():

    model = VAE(hidden_size=HIDDEN_SIZE,   
                latent_size=LATENT_SIZE, 
                feature_dim=FEATURE_DIM, 
                max_decode_len=MAX_DECODE_LEN, 
                depth=DEPTHT, 
                encoding_method=ENCODING_METHOD,
                max_nb=MAX_NB).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        batch = tensorize(cur_attr, cur_conn)
        
        model.zero_grad()
        loss, kl_div, wacc, tacc, pred_loss = model(batch, beta, alpha, gamma)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 50.0)  # Gradient clipping
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "loss": loss.item(),
            "pred_loss": pred_loss.item(),
            "kl_div": kl_div.item(),
        })
        if(epoch % 50 == 0):
            print(f"Epoch {epoch}: Loss={loss.item(): .4f}, Pred Acc={wacc}, Stop Acc={tacc}, PredLoss={pred_loss.item(): .4f}, KL Divergence={kl_div.item(): .4f}")
        
    wandb.save('trained_model.pth', name='model', type='model')
    print("Model saved after epoch", epoch)
    wandb.finish()

def test_decoder():

    model = VAE(hidden_size=HIDDEN_SIZE, 
                latent_size=LATENT_SIZE, 
                feature_dim=FEATURE_DIM, 
                max_decode_len=MAX_DECODE_LEN, 
                depth=DEPTHT, 
                encoding_method=ENCODING_METHOD, 
                max_nb=MAX_NB).to(device)
    
    model.load_state_dict(torch.load('trained_model.pth'))

    batch = tensorize(cur_attr, cur_conn) 
    _, jtenc_holder = batch
    res = model.encoder.encode(jtenc_holder)
    tree_vecs = res[0]

    z_tree_vecs, _ = model.encoder.rsample(z_vecs=tree_vecs)
    for i in range(0,5):
        z_single = z_tree_vecs[i:i+1]
        model.decode(z_single, prob_decode=False)        

if __name__ == "__main__":       
      train_loop()
      test_decoder()
