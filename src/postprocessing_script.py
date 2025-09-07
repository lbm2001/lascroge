import numpy as np
from preprocessing.robo_graph import RoboGraph
import xml.etree.ElementTree as ET
from lxml import etree


file = "/Users/lukasmueller/github/lascroge/robots/locomotion_robots/unitree_go2/go2.xml"
config = "/Users/lukasmueller/github/lascroge/src/preprocessing/feature_conf.yml"

rg = RoboGraph(model_xml_path=file, feature_conf_path=config)
rg.build()


class XmlTreeBuilder():

    def __init__(self, graph):
        self.graph = graph


    def _build_xml_element(self):
        # TODO Implement function to assign the properties to the xml element (body/joint)
        pass


    def _build_body_hierarchy(self, current_node, parent_xml, visited):
        if current_node in visited:
            return
        visited.add(current_node)
        
        node_type = current_node.split("_")[0]
        
        # TODO Read from features
        if node_type == "body":

            body = etree.SubElement(parent_xml, "body")
            # body = build_xml_element(body)
            body.set("name", current_node)
            
            for neighbor in self.graph.neighbors(current_node):
                # TODO Read from features
                if neighbor not in visited and neighbor.startswith("joint"):
                    joint = etree.SubElement(body, "joint")
                    joint.set("name", neighbor)
                    visited.add(neighbor)
                    
                    # TODO Read from features
                    for next_neighbor in self.graph.neighbors(neighbor):
                        if next_neighbor not in visited and next_neighbor.startswith("body"):
                            self.build_body_hierarchy(next_neighbor, body, visited)
    
    def build(self):
        # Create root structure
        root_node = list(self.graph.nodes)[0]
        mujoco = etree.Element("mujoco")
        worldbody = etree.SubElement(mujoco, "worldbody")
        
        self._build_body_hierarchy(root_node, worldbody, set())
        return mujoco


class XmlSaver():

    def __init__(self, xml_root):
        self.xml_root = xml_root

    def save(self, path):
        with open("robot.xml", "wb") as f:
            f.write(etree.tostring(self.xml_root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))


