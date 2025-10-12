import networkx as nx
import mujoco
from pathlib import Path
import os
import numpy as np
import logging
import yaml
from typing import Tuple
import pandas as pd
from preprocessing.feature_matrix_builder import FeatureMatrixBuilder


class NoValidGraphException(Exception):
    def __init__(self, message="Graph is not a valid tree structure"):
        self.message = message
        super().__init__(self.message)


class RoboGraph(nx.Graph):
    """
    Directed graph representation of a MuJoCo model's joint hierarchy.

    Nodes represent joints (by index), with a 'name' attribute.
    Edges connect each joint to all joints of its parent body.
    """
    def __init__(self, model_xml_path: str, feature_conf_path: str):
        super().__init__()
        # Normalize incoming paths so relative inputs work regardless of cwd
        xml_path = Path(model_xml_path).expanduser()
        if not xml_path.is_absolute():
            xml_path = (Path.cwd() / xml_path).resolve()
        else:
            xml_path = xml_path.resolve()

        feature_conf_path = Path(feature_conf_path).expanduser()
        if not feature_conf_path.is_absolute():
            feature_conf_path = (Path.cwd() / feature_conf_path).resolve()
        else:
            feature_conf_path = feature_conf_path.resolve()

        # Build spec and model
        xml_dir = xml_path.parent
        
        # Change to XML directory to resolve relative asset paths
        original_dir = os.getcwd()
        os.chdir(xml_dir)
        
        try:
            xml = xml_path.read_text()
            self.spec = mujoco.MjSpec.from_string(xml)
            self.model = self.spec.compile()
        finally:
            # Always restore original directory
            os.chdir(original_dir)

        # Build namespaces for joints and bodies
        self.jnt_namespace = {i: f"joint_{i}_{self.model.joint(i).name}" for i in range(self.model.njnt)}
        self.body_namespace = {i: f"body_{i}_{self.model.body(i).name}" for i in range(self.model.nbody)}

        self.feature_builder = FeatureMatrixBuilder(model_xml_path=str(xml_path), feature_conf_path=str(feature_conf_path))
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
            for b_id in range(1, self.model.nbody)
        }

        return body_joints


    def build_adj_data(self) -> None:
        body_joints = self.get_body_joints()

        # Add nodes
        for body_id in range(1, self.model.nbody):
            self.add_node(self.body_namespace[body_id], name=self.model.body(body_id).name, type="body")

        for joint_id in range(self.model.njnt):
            self.add_node(self.jnt_namespace[joint_id], name=self.model.joint(joint_id).name, type="joint")

        for body_id, joints in body_joints.items():
            parent_body_id = self.model.body_parentid[body_id]
            for joint_id in joints:
                if parent_body_id != 0:
                    self.add_edge(self.jnt_namespace[joint_id], self.body_namespace[parent_body_id])
                    self.add_edge(self.body_namespace[parent_body_id], self.jnt_namespace[joint_id])

                self.add_edge(self.jnt_namespace[joint_id], self.body_namespace[body_id])
                self.add_edge(self.body_namespace[body_id], self.jnt_namespace[joint_id])

        return self


    def is_tree(self) -> bool:
        """
        Check if the graph is a tree (connected and acyclic).
        """
        if len(self.nodes) == 0:
            return True
        
        # A tree with n nodes has exactly n-1 edges
        if len(self.edges) != len(self.nodes) - 1:
            return False
        
        # Check if the graph is connected
        return nx.is_connected(self)

    def build(self) -> None:
        """
        Builds the model for saving it
        """
        self.build_adj_data()
        self.feature_matrix = self.feature_builder.build_matrix()
        
        # Validate that the graph is a tree
        if not self.is_tree():
            raise NoValidGraphException()
        

    def get_adjacency_matrix(self):
        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")
        return nx.to_numpy_array(self, nodelist=list(self.nodes()))
    

    def get_feature_matrix(self):
        return self.feature_matrix


    def print_adj_matrix(self) -> None:
        """
        Print the adjacency matrix of the graph with node names as labels.
        """

        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")

        # Get the list of nodes and their names
        nodes = list(self.nodes())
        node_names = [self.nodes[node].get("name", str(node)) for node in nodes]

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
