import networkx as nx
import mujoco
from pathlib import Path
import os
import numpy as np
import logging

class RoboGraph(nx.DiGraph):
    """
    Directed graph representation of a MuJoCo model's joint hierarchy.

    Nodes represent joints (by index), with a 'name' attribute.
    Edges connect each joint to all joints of its parent body.
    """
    def __init__(self, xml_path: str):
        super().__init__()
        self.robot_name = os.path.splitext(os.path.basename(xml_path))[0] # Get filename without extension
        xml = Path(xml_path).read_text()
        self.spec = mujoco.MjSpec.from_string(xml)
        self.model = self.spec.compile()

    def build(self) -> None:
        """
        Populate the graph: for each non-root joint, add edges to all joints of its parent body (skips the world "body").
        """
        # Precompute which joints belong to each body
        body_joints = {
            b_id: range(
                self.model.body_jntadr[b_id],
                self.model.body_jntadr[b_id] + self.model.body_jntnum[b_id]
            )
            for b_id in range(self.model.nbody)
        }

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
    

    def save(self, save_dir: str) -> None:
        """
        Safes the adjacency matrix of the robot to the specified location.
        """

        if len(self.nodes) < 1:
            raise Exception("Graph was not yet built.")

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



            