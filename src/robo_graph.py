import networkx as nx
import mujoco
from pathlib import Path
import os
import numpy as np
import logging
import yaml
from typing import Tuple

class RoboGraph(nx.DiGraph):
    """
    Directed graph representation of a MuJoCo model's joint hierarchy.

    Nodes represent joints (by index), with a 'name' attribute.
    Edges connect each joint to all joints of its parent body.
    """
    def __init__(self, model_xml_path: str, conf_path: str):
        super().__init__()
        self.robot_name = os.path.splitext(os.path.basename(model_xml_path))[0] # Get filename without extension
        
        with open(conf_path, "r") as file:
            self.conf = yaml.safe_load(file)

        # Build spec and model
        xml = Path(model_xml_path).read_text()
        self.spec = mujoco.MjSpec.from_string(xml)
        self.model = self.spec.compile()
        
        # Attribute for the feature data
        self.joint_features = None
        self.body_features = None

        
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

    def extract_feature_data(self) -> Tuple[list, list]:
        """
        Extracts features specified in the configuration yaml from the model and returns them.
        """
        joint_feat_names = self.conf["joint_features"]
        body_feat_names = self.conf["body_features"]

        joint_features = []
        body_features = []

        # Extract joint features
        for joint_id in range(self.model.njnt):
            joint = self.model.joint(joint_id)
            feats = []

            for feature_name in joint_feat_names: 
                if hasattr(joint, feature_name):
                    feature_value = getattr(joint, feature_name)
                    feats.append(feature_value)
                else:
                    logging.warning(f"Feature '{feature_name}' not found for joint {joint.name}")
            
            joint_features.append(feats)
        
        # Extract link features
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            feats = []

            for feature_name in body_feat_names: 
                if hasattr(body, feature_name):
                    feature_value = getattr(body, feature_name)
                    feats.append(feature_value)
                else:
                    logging.warning(f"Feature '{feature_name}' not found for body {body.name}")
            
            body_features.append(feats)
        
        return joint_features, body_features


    def transform_feature_data(self, joint_features: list, body_features: list) -> Tuple[np.array, np.array]:
        """
        Transforms the feature data of various types to make it usable for the autoencoder 
        """
        return None, None #TODO: replace with actual logic


    def build_feature_data(self) -> None:
        """
        Builds the features for joints and bodies and saves them in the class attribute.
        """ 
        joint_features_raw, body_features_raw = self.extract_feature_data()
        joint_features_transformed, body_features_transformed = self.transform_feature_data(joint_features_raw, body_features_raw)
        
        self.joint_features = joint_features_transformed
        self.body_features = body_features_transformed
        return self


    def build_adj_data(self) -> None:
        """
        Populate the graph: for each non-root joint, add edges to all joints of its parent body (skips the world "body").
        """
        body_joints = self.get_body_joints()

        for joint_id, body_id in enumerate(self.model.jnt_bodyid):
            parent_body = self.model.body_parentid[body_id]
            if parent_body == 0:
                continue

            joint_name = self.model.joint(joint_id).name
            self.add_node(joint_id, name=joint_name)

            for pbody_joint in body_joints[parent_body]:
                parent_joint_name = self.model.joint(pbody_joint).name
                self.add_node(pbody_joint, name=parent_joint_name)
                self.add_edge(joint_id, pbody_joint)
        return self
    

    def build(self) -> None:
        """
        Builds the model for saving it
        """
        self.build_feature_data()
        self.build_adj_data()


    def save(self, save_dir: str) -> None:
        """
        Safes the adjacency matrix of the robot to the specified location.
        """

        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")

        if self.joint_features == None:
            logging.warning("There are no joint features yet. The data will be saved anyway.")
        
        if self.body_features == None:
            logging.warning("There are no body features yet. The data will be saved anyway.")


        p = Path(save_dir)
        if p.suffix:
            save_path = p
            save_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
            save_path = p / f"{self.robot_name}.npy"

        nodes = list(self.nodes())
        adjacency_matrix = nx.to_numpy_array(self, nodelist=nodes)

        np.save(str(save_path), adjacency_matrix)
        logging.info(f"Adjacency matrix saved to {save_path}")