from robo_graph import RoboGraph
rg = RoboGraph(model_xml_path="go1.xml", conf_path="feature_conf.yml")
rg.build()
rg.save("data/go1_output.npy")
print("Joint features:", rg.joint_features)
print("Body features:", rg.body_features)