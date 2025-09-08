import numpy as np
from preprocessing.robo_graph import RoboGraph
from preprocessing.feature_processor import FeatureProcessor
import yaml
from dm_control import mjcf


class MjcfBuilder:

    def __init__(self, graph, feature_config_path):
        self.graph = graph
        self.feature_map = self._process_feature_config(feature_config_path)
        self.feature_processor = FeatureProcessor()
        self.model = mjcf.RootElement()

    def _process_feature_config(self, feature_config_path):
        feature_map = {}

        with open(feature_config_path, "r") as file:
            conf = yaml.safe_load(file)

        feature_index = 1
        for feat_type in conf.keys():
            feature_map[feat_type] = {}
            for feat in conf[feat_type]:
                feature_name = feat["name"]
                feature_len = feat["len"]
                processing = feat["process"]
                feature_map[feat_type][feature_name] = (
                    feature_index,
                    feature_index + feature_len,
                    processing,
                )
                feature_index += feature_len
        return feature_map

    def _get_attributes(self, feature_map, feature_values):
        attributes = {}
        for feature in feature_map.keys():
            feat_start, feat_end, feat_processing = feature_map[feature]
            attributes[feature] = self.feature_processor.process(
                feature_values[feat_start:feat_end], feat_processing, inv=True
            )
        return attributes

    def _build_element(self, parent, node_name):
        feature_values = self.graph.nodes[node_name]["features"]

        geom_attributes = None
        if feature_values[0] >= 0.5:
            # body node
            node_type = "body"
            element_feature_map = self.feature_map["body_features"]
            geom_feature_map = self.feature_map["geom_features"]
            geom_attributes = self._get_attributes(geom_feature_map, feature_values)
        else:
            # joint node
            node_type = "joint"
            element_feature_map = self.feature_map["joint_features"]

        attributes = self._get_attributes(
            feature_map=element_feature_map, feature_values=feature_values
        )

        if node_type == "body":
            body = parent.add("body", name=node_name, **attributes)
            if geom_attributes:
                body.add("geom", **geom_attributes)
            return body
        else:  # joint
            return parent.add("joint", name=node_name, **attributes)

    def _build_hierarchy(self, current_node, parent, visited):
        if current_node in visited:
            return
        visited.add(current_node)

        node_type = current_node.split("_")[0]
        if node_type == "body":
            body = self._build_element(parent, current_node)
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited and neighbor.startswith("joint"):
                    joint = self._build_element(body, neighbor)
                    visited.add(neighbor)
                    for next_neighbor in self.graph.neighbors(neighbor):
                        if (
                            next_neighbor not in visited
                            and next_neighbor.startswith("body")
                        ):
                            self._build_hierarchy(next_neighbor, body, visited)

    def build(self):
        root_node = list(self.graph.nodes)[0]
        worldbody = self.model.worldbody
        self._build_hierarchy(root_node, worldbody, set())
        return self.model


class MjcfSaver:
    def __init__(self, mjcf_model):
        self.mjcf_model = mjcf_model

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.mjcf_model.to_xml_string())


# Usage
file = "/Users/lukasmueller/github/lascroge/robots/locomotion_robots/unitree_go2/go2.xml"
config = "/Users/lukasmueller/github/lascroge/src/preprocessing/feature_conf.yml"

rg = RoboGraph(model_xml_path=file, feature_conf_path=config)
rg.build()

# Assign features to node (dummy initialization for now)
for i, node in enumerate(rg.nodes):
    rg.nodes[node]["features"] = rg.feature_matrix[i]

mjcf_builder = MjcfBuilder(rg, feature_config_path=config)
mjcf_model = mjcf_builder.build()

saver = MjcfSaver(mjcf_model)
saver.save("./robot.xml")
