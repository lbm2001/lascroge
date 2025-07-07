import argparse
import glob
import os
import logging

from robo_graph import RoboGraph
from graph_saver import GraphSaver

# Command for copy-pasting: python src/convert_mujoco_xml.py -i "/Users/lukasmueller/github/lascroge/data/mujoco_models" -s "/Users/lukasmueller/github/lascroge/data/robot_graphs" -c "/Users/lukasmueller/github/lascroge/src/feature_conf.yml"

def main():
    parser = argparse.ArgumentParser(prog="convert_mujoco_xml.py")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Directory with mujoco files"
    )
    parser.add_argument(
        "-c", "--config",
        help="Configuration for the features"
    )
    parser.add_argument(
        "-s", "--save",
        default=".",
        help="Directory to save the data in"
    )

    args = parser.parse_args()

    gs = GraphSaver()
    xml_files = glob.glob(os.path.join(args.input, "*.xml"))
    if not xml_files:
        raise Exception(f"No XML files found in directory: {args.input}")
    
    for file in xml_files:
        logging.info(f"Processing {file} ...")
        rg = RoboGraph(model_xml_path=file, feature_conf_path=args.config)
        rg.build()
        adj = rg.get_adjacency_matrix()
        feat = rg.get_feature_matrix()
        gs.add_graph(adj, feat)
    
    gs.save(args.save)

if __name__ == "__main__":
    main()