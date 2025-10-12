# lascroge

This repository experiments with learning generative models of robot morphologies. MuJoCo XML descriptions are converted into graph representations (adjacency + feature matrices), a variational autoencoder (VAE) is trained on those graphs, and decoded samples can be turned back into MuJoCo XML files.

## Setup

1. Use Python 3.10+ and create a virtual environment.
2. Install the core dependencies: `pip install torch mujoco numpy networkx lxml pyyaml wandb`.
3. If you plan to log to Weights & Biases, run `wandb login`. Otherwise set `WANDB_MODE=offline` before training.

The MuJoCo Python bindings expect the MuJoCo binaries to be installed and the `MUJOCO_GL` environment variable to be set appropriately for your platform.

## Preprocessing: XML → Graph Dataset

Running `src/convert_mujoco_xml.py` walks a directory of MuJoCo models, extracts tree-structured body/joint graphs, and stores them as NumPy arrays.

```bash
python src/convert_mujoco_xml.py \
  -i robots/locomotion_robots \
  -c src/preprocessing/feature_conf.yml \
  -s data/robot_graphs
```

- `-i` points to a folder that contains subdirectories with `.xml` models.
- `-c` is the feature definition file that controls how body, geom, inertial, and joint attributes are encoded.
- `-s` is where the script writes `adj.npy` and `feat.npy` (object arrays so each graph can have a different size).

Inspect the resulting `.npy` files to make sure the feature dimension matches the VAE configuration (`feature_dim`).

## Training the VAE

1. Edit `src/train_config.yml` (or copy it) to point to the dataset you just produced and choose where to store the model weights.
2. Adjust the hyperparameters if needed. In particular, set `feature_dim` to the width of your feature vectors.
3. Launch training:

   ```bash
   python src/training.py
   ```

The script loads the config file, normalizes features per node type, and streams metrics to Weights & Biases. A trained model checkpoint is saved to `model_save_path`, along with `_norm_params.npy` that holds the feature normalization statistics.

## Decoding and Rebuilding XML

1. To sample from a trained model and inspect the decoded graph, call the helper in `training.py`:

   ```bash
   python -c "from src.training import test_decoder; test_decoder('models/go2_test/model.pth')"
   ```

   Update the path to match your checkpoint. The script prints the denormalized features of the decoded root node and reports the number of nodes generated.

2. To turn decoded features into a MuJoCo XML file, use `src/postprocessing_script.py`:
   - Provide a template MuJoCo XML and the same feature config used during preprocessing (edit the `file` and `config` variables at the bottom of the script or wrap it in your own entry point).
   - Before calling `XmlTreeBuilder`, assign the VAE-generated feature vectors to `rg.nodes[node]["features"]` in the order of `rg.nodes`.
   - Run `python src/postprocessing_script.py` to emit a new XML file (`robot.xml` by default).

This end-to-end process lets you go from raw MuJoCo assets → graph dataset → trained latent model → decoded robot designs in MuJoCo format.
