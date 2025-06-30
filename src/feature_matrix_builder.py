import yaml
import mujoco
import numpy as np
import logging
from pathlib import Path
from feature_processor import FeatureProcessor


class FeatureMatrixBuilder:
    """
    Builds a complete feature matrix (as numpy array) from a Mujoco model XML,
    using a configuration file and the FeatureProcessor.
    """

    def __init__(self, model_xml_path: str, feature_conf_path: str):
        with open(feature_conf_path, "r") as file:
            self.conf = yaml.safe_load(file)

        # Build spec and model
        xml = Path(model_xml_path).read_text()
        self.spec = mujoco.MjSpec.from_string(xml)
        self.model = self.spec.compile()

        self.processor = FeatureProcessor()


    def build_matrix(self):
        all_features = []

        # Process joint features
        for joint_id in range(self.model.njnt):
            joint = self.model.joint(joint_id)
            feats = [1.0] #is_joint flag
            feats.extend(self._extract_entity_features(joint, self.conf["joint_features"]))
            all_features.append(feats)

        # Process body features
        for body_id in range(self.model.nbody):
            body = self.model.body(body_id)
            feats = [0.0] # is_joint flag
            feats.extend(self._extract_entity_features(body, self.conf["body_features"]))
            all_features.append(feats)

        # Pad features to generate matrix
        max_len = max(len(f_vec) for f_vec in all_features)
        all_features_padded = [f + [0.0] * (max_len - len(f)) for f in all_features]

        return np.array(all_features_padded, dtype=np.float32)

    def _extract_entity_features(self, entity, feature_list_config):
        feats = []
        for feature_entry in feature_list_config:
            feat_name = feature_entry["name"]
            method = feature_entry.get("process", "identity")

            if hasattr(entity, feat_name):
                raw_value = getattr(entity, feat_name)
            else:
                logging.warning(f"Feature '{feat_name}' not found in {entity}.")
                raw_value = None

            processed = self.processor.process(raw_value, method)
            feats.extend(processed)

        return feats
    

builder = FeatureMatrixBuilder("./data/mujoco_models/simple_robot.xml", "src/feature_conf.yml")
builder.build_matrix()