import numpy as np
from preprocessing.robo_graph import RoboGraph
from preprocessing.feature_processor import FeatureProcessor
from lxml import etree
import yaml


class XmlTreeBuilder:
    def __init__(self, graph, feature_config_path):
        self.graph = graph
        self.feature_map = self._process_feature_config(feature_config_path)
        self.feature_processor = FeatureProcessor()
        self.inertial_feature_map = {"iquat": "quat", "ipos": "pos", "inertia": "diaginertia"}

    def _process_feature_config(self, feature_config_path):
        feature_map = {}
        with open(feature_config_path, "r") as file:
            conf = yaml.safe_load(file)

        feature_index = 0
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
            feat_start = feature_map[feature][0]
            feat_end = feature_map[feature][1]
            feat_processing = feature_map[feature][2]

            feature_name = feature
            if feature in self.inertial_feature_map.keys():
                feature_name = self.inertial_feature_map[feature]

            attributes[feature_name] = self.feature_processor.process(
                feature_values[feat_start:feat_end], feat_processing, inv=True
            )
        return attributes

    def _build_xml_element(self, parent_xml, node_name):
        feature_values = self.graph.nodes[node_name]["features"]

        element = None

        if feature_values[0] >= 0.5:
            node_type = "body"
            body_feature_map = self.feature_map["body_features"]
            geom_feature_map = self.feature_map["geom_features"]
            inertial_feature_map = self.feature_map["inertial_features"]

            body_attributes = self._get_attributes(body_feature_map, feature_values)
            geom_attributes = self._get_attributes(geom_feature_map, feature_values)
            inertial_attributes = self._get_attributes(inertial_feature_map, feature_values)

            element = etree.SubElement(parent_xml, node_type)
            inertial = etree.SubElement(element, "inertial")
            geom = etree.SubElement(element, "geom")

            element.attrib.update(body_attributes)
            element.set("name", node_name)
            geom.attrib.update(geom_attributes)
            inertial.attrib.update(inertial_attributes)

        else:
            node_type = "joint"
            joint_feature_map = self.feature_map["joint_features"]
            joint_attributes = self._get_attributes(
                feature_map=joint_feature_map, feature_values=feature_values
            )
            element = etree.SubElement(parent_xml, node_type)
            element.set("name", node_name)
            element.attrib.update(joint_attributes)
                

        return element

    def _build_body_hierarchy(self, current_node, parent_xml, visited):
        if current_node in visited:
            return
        visited.add(current_node)

        node_type = current_node.split("_")[0]  # TODO: read from features
        if node_type == "body":
            body = self._build_xml_element(parent_xml=parent_xml, node_name=current_node)

            for joint_neighbor in self.graph.neighbors(current_node):
                if joint_neighbor not in visited and joint_neighbor.startswith("joint"):
                    visited.add(joint_neighbor)

                    for next_neighbor in self.graph.neighbors(joint_neighbor):
                        if next_neighbor not in visited and next_neighbor.startswith("body"):
                            # Create the next body first
                            next_body = self._build_xml_element(parent_xml=body, node_name=next_neighbor)
                            # Then create the joint as a child of that body
                            _ = self._build_xml_element(parent_xml=next_body, node_name=joint_neighbor)
                            
                            # Continue the recursion
                            self._build_body_hierarchy(next_neighbor, body, visited)

    def build(self):
        # Create root structure
        root_node = list(self.graph.nodes)[0]
        mujoco = etree.Element("mujoco")
        worldbody = etree.SubElement(mujoco, "worldbody")
        self._build_body_hierarchy(root_node, worldbody, set())
        return mujoco


class XmlSaver:
    def __init__(self, xml_root):
        self.xml_root = xml_root

    def save(self, path):
        with open(path, "wb") as f:
            f.write(
                etree.tostring(
                    self.xml_root, pretty_print=True, xml_declaration=True, encoding="UTF-8"
                )
            )


if __name__ == "__main__":
    file = "/Users/lukasmueller/github/lascroge/robots/locomotion_robots/unitree_go2/go2.xml"
    config = "/Users/lukasmueller/github/lascroge/src/preprocessing/feature_conf.yml"

    rg = RoboGraph(model_xml_path=file, feature_conf_path=config)
    rg.build()

    # Assign features to nodes (TODO: remove when using real VAE output)
    for i, node in enumerate(rg.nodes):
        rg.nodes[node]["features"] = rg.feature_matrix[i]

    xml_builder = XmlTreeBuilder(rg, feature_config_path=config)
    xml_root = xml_builder.build()

    xml_saver = XmlSaver(xml_root=xml_root)
    xml_saver.save("./robot.xml")
