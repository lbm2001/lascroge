import yaml
import mujoco
import numpy as np
import logging
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from preprocessing.feature_processor import FeatureProcessor


class FeatureMatrixBuilder:
    """
    Builds a complete feature matrix (as numpy array) from a Mujoco model XML,
    using a configuration file and the FeatureProcessor.
    """

    def __init__(self, 
                 model_xml_path: str, 
                 feature_conf_path: str):
        with open(feature_conf_path, "r") as file:
            self.conf = yaml.safe_load(file)

        # Build spec and model
        xml_path = Path(model_xml_path)
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

        self.processor = FeatureProcessor()

        # Geom map
        self.geom_map = {
            "type": self.model.geom_type, 
            "size": self.model.geom_size, 
            "pos": self.model.geom_pos, 
            "quat": self.model.geom_quat
            }
        
        # Scaler
        self.scaler = StandardScaler()


    def build_matrix(self):
        all_features = []

        # Process joint features
        for joint_id in range(self.model.njnt):
            joint = self.model.joint(joint_id)
            feats = [1.0] #is_joint flag
            feats.extend(self._extract_entity_features(joint, self.conf["joint_features"]))
            all_features.append(feats)

        # Process body features
        for body_id in range(1, self.model.nbody):
            body = self.model.body(body_id)
            feats = [0.0] # is_joint flag
            feats.extend(self._extract_entity_features(body, self.conf["body_features"]))
            
            # TODO: For now, only retrieve the first primitive geom
            first_primitive_geom_id = self._get_first_primitive_geom_id(body_id)

            if first_primitive_geom_id is not None:
                feats.extend(self._extract_geom_features(first_primitive_geom_id, self.conf["geom_features"]))

            #for id in primitive_geoms:
            #    feats.extend(self._extract_geom_features(id, self.conf["geom_features"]))

            all_features.append(feats)

        # Pad features to generate matrix
        max_len = max(len(f_vec) for f_vec in all_features)
        all_features_padded = [f + [0.0] * (max_len - len(f)) for f in all_features] 

        feature_matrix = np.array(all_features_padded, dtype=np.float32)
        feature_matrix_normalized = self._normalize(self.scaler, feature_matrix)

        return feature_matrix_normalized


    def _normalize(self, scaler, feature_matrix):
        return scaler.fit_transform(feature_matrix)


    def _extract_entity_features(self, entity, feature_list_config):
        feats = []
        for feature_entry in feature_list_config:
            feat_name = feature_entry["name"]
            method = feature_entry.get("process", "identity")

            if hasattr(entity, feat_name):
                raw_value = getattr(entity, feat_name)
            else:
                logging.warning(f"Feature '{feat_name}' not found in {entity}. Using default value")
                raw_value = None

            processed = self.processor.process(raw_value, method)
            feats.extend(processed)

        return feats
    

    def _extract_geom_features(self, geom_id, feature_list_config):
        feats = []
        for feature_entry in feature_list_config:
            feat_name = feature_entry["name"]
            method = feature_entry.get("process", "identity")
            raw_value = self.geom_map.get(feat_name)[geom_id]
            processed = self.processor.process(raw_value, method)

            feats.extend(processed)
        
        return feats
    

    def _get_primitive_geom_ids(self, body_id):
        # Get all geoms for this body
        geom_ids = [i for i in range(self.model.ngeom) 
                    if self.model.geom_bodyid[i] == body_id]
        
        # Filter out mesh geoms
        return [geom_id for geom_id in geom_ids 
                if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH]


    def _get_first_primitive_geom_id(self, body_id):
        # Get all geoms for this body
        geom_ids = [i for i in range(self.model.ngeom) 
                    if self.model.geom_bodyid[i] == body_id]

        # Find the first primitive (non-mesh) geom
        for geom_id in geom_ids:
            if self.model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                return geom_id
        
        # Return None if no primitive geom found
        return None