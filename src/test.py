import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Tuple, Optional
import mujoco


class MujocoXMLGenerator:
    """
    Converts decoded graph structure (TreeNodes) back into a MuJoCo XML file.
    """
    
    def __init__(self):
        # Feature indices based on the configuration
        self.body_feature_indices = {
            'is_joint': 0,
            'pos': (1, 4),  # 3 values
            'mass': 4,
            'inertia': (5, 8),  # 3 values 
            'quat': (8, 12),  # 4 values
            'subtreemass': 12,
            'ipos': (13, 16),  # 3 values
            'iquat': (16, 20),  # 4 values
            # Geom features start after body features
            'geom_size': (20, 23),  # 3 values
            'geom_pos': (23, 26),  # 3 values
            'geom_quat': (26, 30)  # 4 values
        }
        
        self.joint_feature_indices = {
            'is_joint': 0,
            'type': 1,
            'range': (2, 4),  # 2 values
            'stiffness': 4,
            'axis': (5, 8),  # 3 values
            'pos': (8, 11)  # 3 values
        }
        
    def generate_xml(self, root_node, all_nodes: List, output_path: str = None) -> str:
        """
        Generate MuJoCo XML from decoded tree structure.
        
        Args:
            root_node: Root TreeNode
            all_nodes: List of all TreeNodes
            output_path: Optional path to save XML file
            
        Returns:
            XML string
        """
        # Build adjacency relationships
        self._build_node_relationships(all_nodes)
        
        # Create XML structure
        mujoco_root = ET.Element("mujoco", model="generated_robot")
        
        # Add compiler and option sections with defaults
        compiler = ET.SubElement(mujoco_root, "compiler", angle="radian", autolimits="true")
        option = ET.SubElement(mujoco_root, "option", cone="elliptic", impratio="100")
        
        # Add default settings
        self._add_defaults(mujoco_root)
        
        # Add worldbody
        worldbody = ET.SubElement(mujoco_root, "worldbody")
        
        # Find the trunk/root body (should be the first body node)
        trunk_node = self._find_trunk_node(all_nodes)
        
        if trunk_node:
            # Build hierarchical structure starting from trunk
            self._build_body_hierarchy(trunk_node, worldbody, all_nodes)
        
        # Add actuators for all joints
        self._add_actuators(mujoco_root, all_nodes)
        
        # Convert to pretty XML string
        xml_string = self._prettify_xml(mujoco_root)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(xml_string)
        
        return xml_string
    
    def _build_node_relationships(self, all_nodes: List):
        """Ensure all nodes have proper idx attributes."""
        for i, node in enumerate(all_nodes):
            if not hasattr(node, 'idx'):
                node.idx = i
    
    def _find_trunk_node(self, all_nodes: List):
        """Find the trunk/root body node (first body in the list)."""
        for node in all_nodes:
            features = node.features.cpu().numpy() if hasattr(node.features, 'cpu') else node.features
            if features[0] == 0:  # is_body
                return node
        return None
    
    def _is_joint_node(self, node) -> bool:
        """Check if a node is a joint."""
        features = node.features.cpu().numpy() if hasattr(node.features, 'cpu') else node.features
        return features[0] == 1
    
    def _is_body_node(self, node) -> bool:
        """Check if a node is a body."""
        features = node.features.cpu().numpy() if hasattr(node.features, 'cpu') else node.features
        return features[0] == 0
    
    def _extract_body_features(self, node) -> Dict:
        """Extract body features from node."""
        features = node.features.cpu().numpy() if hasattr(node.features, 'cpu') else node.features
        
        idx = self.body_feature_indices
        return {
            'pos': features[idx['pos'][0]:idx['pos'][1]],
            'mass': float(features[idx['mass']]),
            'inertia': features[idx['inertia'][0]:idx['inertia'][1]],
            'quat': features[idx['quat'][0]:idx['quat'][1]],
            'subtreemass': float(features[idx['subtreemass']]),
            'ipos': features[idx['ipos'][0]:idx['ipos'][1]],
            'iquat': features[idx['iquat'][0]:idx['iquat'][1]],
            'geom_size': features[idx['geom_size'][0]:idx['geom_size'][1]],
            'geom_pos': features[idx['geom_pos'][0]:idx['geom_pos'][1]],
            'geom_quat': features[idx['geom_quat'][0]:idx['geom_quat'][1]]
        }
    
    def _extract_joint_features(self, node) -> Dict:
        """Extract joint features from node."""
        features = node.features.cpu().numpy() if hasattr(node.features, 'cpu') else node.features
        
        idx = self.joint_feature_indices
        return {
            'type': int(features[idx['type']]),
            'range': features[idx['range'][0]:idx['range'][1]],
            'stiffness': float(features[idx['stiffness']]),
            'axis': features[idx['axis'][0]:idx['axis'][1]],
            'pos': features[idx['pos'][0]:idx['pos'][1]]
        }
    
    def _build_body_hierarchy(self, body_node, parent_element, all_nodes: List, visited=None):
        """Recursively build body hierarchy."""
        if visited is None:
            visited = set()
        
        if body_node.idx in visited:
            return
        
        visited.add(body_node.idx)
        
        # Extract body features
        body_features = self._extract_body_features(body_node)
        
        # Create body element
        body_name = f"body_{body_node.idx}"
        body_elem = ET.SubElement(parent_element, "body", name=body_name)
        
        # Set body attributes
        self._set_vector_attribute(body_elem, "pos", body_features['pos'])
        self._set_vector_attribute(body_elem, "quat", body_features['quat'])
        
        # Add inertial properties
        inertial = ET.SubElement(body_elem, "inertial")
        inertial.set("mass", str(body_features['mass']))
        self._set_vector_attribute(inertial, "diaginertia", body_features['inertia'][:3])
        self._set_vector_attribute(inertial, "pos", body_features['ipos'][:3])
        self._set_vector_attribute(inertial, "quat", body_features['iquat'][:4])
        
        # Add geom (collision/visual geometry)
        self._add_geom(body_elem, body_features, body_name)
        
        # Find connected joints and child bodies
        for neighbor in body_node.neighbors:
            if neighbor.idx in visited:
                continue
            
            if self._is_joint_node(neighbor):
                # Add joint
                self._add_joint(body_elem, neighbor, body_name)
                
                # Find child body connected through this joint
                for joint_neighbor in neighbor.neighbors:
                    if joint_neighbor.idx != body_node.idx and self._is_body_node(joint_neighbor):
                        self._build_body_hierarchy(joint_neighbor, body_elem, all_nodes, visited)
            
            elif self._is_body_node(neighbor):
                # Direct body-to-body connection (less common)
                self._build_body_hierarchy(neighbor, body_elem, all_nodes, visited)
    
    def _add_joint(self, parent_body_elem, joint_node, parent_body_name: str):
        """Add a joint to a body."""
        joint_features = self._extract_joint_features(joint_node)
        
        joint_name = f"joint_{joint_node.idx}_{parent_body_name}"
        joint_elem = ET.SubElement(parent_body_elem, "joint", name=joint_name)
        
        # Set joint type
        joint_type_map = {0: "hinge", 1: "slide", 2: "ball", 3: "free"}
        joint_type = joint_type_map.get(joint_features['type'], "hinge")
        joint_elem.set("type", joint_type)
        
        # Set other attributes
        self._set_vector_attribute(joint_elem, "axis", joint_features['axis'])
        self._set_vector_attribute(joint_elem, "range", joint_features['range'])
        self._set_vector_attribute(joint_elem, "pos", joint_features['pos'])
        
        if joint_features['stiffness'] > 0:
            joint_elem.set("stiffness", str(joint_features['stiffness']))
    
    def _add_geom(self, parent_body_elem, body_features: Dict, body_name: str):
        """Add geometry to a body."""
        geom_elem = ET.SubElement(parent_body_elem, "geom")
        geom_elem.set("name", f"geom_{body_name}")
        
        # Determine geom type based on size parameters
        size = body_features['geom_size']
        if np.sum(size > 0) == 1:  # One non-zero dimension -> sphere
            geom_elem.set("type", "sphere")
            geom_elem.set("size", str(max(size)))
        elif np.sum(size > 0) == 2:  # Two non-zero dimensions -> cylinder or capsule
            geom_elem.set("type", "cylinder")
            self._set_vector_attribute(geom_elem, "size", size[:2])
        else:  # Three dimensions -> box
            geom_elem.set("type", "box")
            self._set_vector_attribute(geom_elem, "size", size)
        
        # Set position and orientation
        if np.any(body_features['geom_pos'] != 0):
            self._set_vector_attribute(geom_elem, "pos", body_features['geom_pos'])
        
        if not np.allclose(body_features['geom_quat'], [1, 0, 0, 0]):
            self._set_vector_attribute(geom_elem, "quat", body_features['geom_quat'])
    
    def _add_defaults(self, mujoco_root):
        """Add default settings to the model."""
        default = ET.SubElement(mujoco_root, "default")
        default_class = ET.SubElement(default, "default", class_="robot")
        
        geom_default = ET.SubElement(default_class, "geom")
        geom_default.set("friction", "0.6")
        geom_default.set("margin", "0.001")
        
        joint_default = ET.SubElement(default_class, "joint")
        joint_default.set("damping", "2")
        joint_default.set("armature", "0.01")
    
    def _add_actuators(self, mujoco_root, all_nodes: List):
        """Add actuators for all joints."""
        actuator = ET.SubElement(mujoco_root, "actuator")
        
        for node in all_nodes:
            if self._is_joint_node(node):
                # Find parent body to get joint name
                for neighbor in node.neighbors:
                    if self._is_body_node(neighbor):
                        parent_body_name = f"body_{neighbor.idx}"
                        joint_name = f"joint_{node.idx}_{parent_body_name}"
                        
                        position_elem = ET.SubElement(actuator, "position")
                        position_elem.set("name", f"actuator_{joint_name}")
                        position_elem.set("joint", joint_name)
                        position_elem.set("kp", "100")
                        break
    
    def _set_vector_attribute(self, element, attr_name: str, values):
        """Set a vector attribute with proper formatting."""
        if isinstance(values, (list, np.ndarray)):
            if len(values) > 0:
                value_str = " ".join([str(float(v)) for v in values])
                element.set(attr_name, value_str)
        else:
            element.set(attr_name, str(float(values)))
    
    def _prettify_xml(self, elem) -> str:
        """Return a pretty-printed XML string."""
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


# Example usage
def decode_to_xml(root_node, all_nodes, output_path="generated_robot.xml"):
    """
    Convert decoded graph to MuJoCo XML.
    
    Args:
        root_node: Root TreeNode from decoder
        all_nodes: List of all TreeNodes from decoder
        output_path: Path to save the generated XML
        
    Returns:
        XML string
    """
    generator = MujocoXMLGenerator()
    xml_string = generator.generate_xml(root_node, all_nodes, output_path)
    
    # Optionally validate with MuJoCo
    try:
        # Try to compile the model to check validity
        spec = mujoco.MjSpec.from_string(xml_string)
        model = spec.compile()
        print(f"Successfully generated valid MuJoCo model with {model.nbody} bodies and {model.njnt} joints")
    except Exception as e:
        print(f"Warning: Generated XML may have issues: {e}")
    
    return xml_string