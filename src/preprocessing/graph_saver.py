from pathlib import Path
import numpy as np
import logging

class GraphSaver():

    def __init__(self):
        self.data = []
    
    def add_graph(self, graph, features):
        self.data.append((graph, features))
    
    def save(self, save_dir: str) -> None:
        """
        Saves the data in the provided location.
        """
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)

        save_path_adj_matrix = p / "adj.npy"
        save_path_features = p / "feat.npy"

        adjs = [graph for graph, _ in self.data]
        feats = [feat for _, feat in self.data]

        np.save(str(save_path_adj_matrix), np.array(adjs, dtype=object)) #TODO: Check if this works
        np.save(str(save_path_features), np.array(feats, dtype=object))

        logging.info(f"Data saved in {p}")