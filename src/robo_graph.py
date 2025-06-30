import networkx as nx
import mujoco
from pathlib import Path
import os
import numpy as np
import logging
import yaml
from typing import Tuple
import pandas as pd
from feature_matrix_builder import FeatureMatrixBuilder

class RoboGraph(nx.DiGraph):
    """
    Directed graph representation of a MuJoCo model's joint hierarchy.

    Nodes represent joints (by index), with a 'name' attribute.
    Edges connect each joint to all joints of its parent body.
    """
    def __init__(self, model_xml_path: str, feature_conf_path: str):
        super().__init__()
        self.robot_name = os.path.splitext(os.path.basename(model_xml_path))[0] # Get filename without extension
        
        

        # Build spec and model
        xml = Path(model_xml_path).read_text()
        self.spec = mujoco.MjSpec.from_string(xml)
        self.model = self.spec.compile()

        self.feature_builder = FeatureMatrixBuilder(model_xml_path=model_xml_path, feature_conf_path=feature_conf_path)
        self.feature_matrix = None

        
    def get_body_joints(self) -> None:
        """
        Precompute which joints belong to each body.
        """
        body_joints = {
            b_id: range(
                self.model.body_jntadr[b_id],
                self.model.body_jntadr[b_id] + self.model.body_jntnum[b_id]
            )
            for b_id in range(self.model.nbody)
        }

        return body_joints


    def build_adj_data(self) -> None:
        # TODO: Find nicer representation for joint_ids than * -1
        body_joints = self.get_body_joints()

        # Add nodes
        for body_id in body_joints.keys():
            body_name = self.model.body(body_id).name
            self.add_node(body_id, name=body_name, type="body")

        for joint_id in range(self.model.njnt):
            joint_name = self.model.joint(joint_id).name
            self.add_node(-1 * joint_id, name=joint_name, type="joint")

        for body_id, joints in body_joints.items():

            parent_body = self.model.body_parentid[body_id]
            for joint_id in joints:

                self.add_edge(-1 * joint_id, parent_body)
                self.add_edge(parent_body, -1 * joint_id)

                self.add_edge(-1 * joint_id, body_id)
                self.add_edge(body_id, -1 * joint_id)

        return self


    def build(self) -> None:
        """
        Builds the model for saving it
        """
        self.build_adj_data()
        self.feature_matrix = self.feature_builder.build_matrix()
        


    def save(self, save_dir: str) -> None:
        """
        Safes the adjacency matrix and features of the robot to the specified location.
        """

        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")


        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        save_path_adj_matrix = p / f"{self.robot_name}.npy"
        save_path_features = p / f"{self.robot_name}_features.npy"

        nodes = list(self.nodes())
        adjacency_matrix = nx.to_numpy_array(self, nodelist=nodes)

        np.save(str(save_path_adj_matrix), adjacency_matrix)
        np.save(str(save_path_features), self.feature_matrix)

        logging.info(f"Adjacency matrix and feature matrix saved in {p}")


    def print_adj_matrix(self) -> None:
        """
        Print the adjacency matrix of the graph with node names as labels.
        """

        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")

        # Get the list of nodes and their names
        nodes = list(self.nodes())
        node_names = [self.nodes[node]["name"] for node in nodes]

        # Generate the adjacency matrix
        adjacency_matrix = nx.to_numpy_array(self, nodelist=nodes)

        # Create a pandas DataFrame for better visualization
        adj_df = pd.DataFrame(adjacency_matrix, index=node_names, columns=node_names)

        print("Adjacency Matrix with Node Names:")
        print(adj_df)
    
    def print_pastable_adj_matrix(self) -> None:
        """
        Prints the adjacency matrix ready to paste into: https://graphonline.top
        """
        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")
        
        adjacency_matrix = nx.to_numpy_array(self).astype(int)
        for row in adjacency_matrix:
            print(",".join(map(str, row)))
