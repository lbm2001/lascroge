import numpy as np
from preprocessing.robo_graph import RoboGraph
from preprocessing.feature_processor import FeatureProcessor
from lxml import etree
import mujoco
import yaml

class MuJoCoTreeBuilder():

    def __init__(self, graph, feature_config_path):
        self.graph = graph
        self.feature_map = self._process_feature_config(feature_config_path)
        self.feature_processor = FeatureProcessor()

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

                feature_map[feat_type][feature_name] = (feature_index, feature_index + feature_len, processing)
                feature_index += feature_len

        return feature_map
        
    def _get_attributes(self, feature_map, feature_values):
        attributes = {}

        for feature in feature_map.keys():
            feat_start = feature_map[feature][0]
            feat_end = feature_map[feature][1]
            feat_processing = feature_map[feature][2]

            attributes[feature] = self.feature_processor.process(
                feature_values[feat_start : feat_end], feat_processing, inv=True
            )
        
        return attributes

    def _build_xml_element(self, parent_xml, node_name):
        feature_values = self.graph.nodes[node_name]["features"]
        element_feature_map = None
        geom_attributes = None

        if feature_values[0] >= 0.5:
            node_type = "body"
            element_feature_map = self.feature_map["body_features"]
            geom_feature_map = self.feature_map["geom_features"]
            geom_attributes = self._get_attributes(geom_feature_map, feature_values)

        else: 
            node_type = "joint"   
            element_feature_map = self.feature_map["joint_features"]
             
        attributes = self._get_attributes(feature_map=element_feature_map, feature_values=feature_values)

        element = etree.SubElement(parent_xml, node_type)
        element.set("name", node_name)

        for attr_name, attr_value in attributes.items():
            element.set(attr_name, str(attr_value))

        if geom_attributes:
            geom = etree.SubElement(element, "geom")
            for attr_name, attr_value in geom_attributes.items():
                geom.set(attr_name, str(attr_value))

        return element

    def _build_body_hierarchy(self, current_node, parent_xml, visited):
        if current_node in visited:
            return
        visited.add(current_node)
        
        node_type = current_node.split("_")[0]
        
        if node_type == "body":
            body = self._build_xml_element(parent_xml=parent_xml, node_name=current_node)
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited and neighbor.startswith("joint"):
                    _ = self._build_xml_element(parent_xml=parent_xml, node_name=neighbor)
                    visited.add(neighbor)
                    
                    for next_neighbor in self.graph.neighbors(neighbor):
                        if next_neighbor not in visited and next_neighbor.startswith("body"):
                            self._build_body_hierarchy(next_neighbor, body, visited)
    
    def build_xml(self):
        """Build XML structure using lxml"""
        root_node = list(self.graph.nodes)[0]
        mujoco_element = etree.Element("mujoco")
        worldbody = etree.SubElement(mujoco_element, "worldbody")
        
        self._build_body_hierarchy(root_node, worldbody, set())
        return mujoco_element
    
    def build_mujoco_model(self):
        """Build MuJoCo model from XML"""
        xml_root = self.build_xml()
        xml_string = etree.tostring(xml_root, encoding='unicode', pretty_print=True)
        
        # Create MuJoCo model from XML string
        try:
            model = mujoco.MjModel.from_xml_string(xml_string)
            return model, xml_string
        except Exception as e:
            print(f"Error creating MuJoCo model: {e}")
            print("XML content:")
            print(xml_string)
            return None, xml_string


class MuJoCoModelSaver():

    def __init__(self, model, xml_string):
        self.model = model
        self.xml_string = xml_string

    def save_xml(self, xml_path):
        """Save XML file"""
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(self.xml_string)
    
    def save_binary(self, mjb_path):
        """Save MuJoCo binary model (if model is valid)"""
        if self.model is not None:
            mujoco.mj_saveModel(self.model, mjb_path)
        else:
            raise ValueError("Cannot save binary model: MuJoCo model is None")
    
    def get_model_info(self):
        """Get basic model information"""
        if self.model is None:
            return "Model is None - XML parsing failed"
        
        info = {
            'nbody': self.model.nbody,
            'njnt': self.model.njnt,
            'ngeom': self.model.ngeom,
            'nq': self.model.nq,  # Number of position coordinates
            'nv': self.model.nv,  # Number of velocity coordinates
        }
        return info


# Usage
file = "/Users/lukasmueller/github/lascroge/robots/locomotion_robots/unitree_go2/go2.xml"
config = "/Users/lukasmueller/github/lascroge/src/preprocessing/feature_conf.yml"

rg = RoboGraph(model_xml_path=file, feature_conf_path=config)
rg.build()

# Assign features to nodes
for i, node in enumerate(rg.nodes):
    rg.nodes[node]["features"] = rg.feature_matrix[i]

# Build model
builder = MuJoCoTreeBuilder(rg, feature_config_path=config)
model, xml_string = builder.build_mujoco_model()

# Save and work with model
saver = MuJoCoModelSaver(model, xml_string)
saver.save_xml("./robot.xml")

if model is not None:
    # Save binary model (faster loading)
    saver.save_binary("./robot.mjb")
    
    # Get model info
    print("Model info:", saver.get_model_info())
    
    # Create simulation data
    data = mujoco.MjData(model)
    
    # Run simulation step
    mujoco.mj_step(model, data)
    
    print(f"Model has {model.nbody} bodies and {model.njnt} joints")
else:
    print("Failed to create MuJoCo model - check XML structure")