import argparse
from robo_graph import RoboGraph

def main():
    parser = argparse.ArgumentParser(prog="convert_mujoco_xml.py")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input MuJoCo XML file"
    )
    parser.add_argument(
        "-c", "--config",
        help="Configuration for the features"
    )
    parser.add_argument(
        "-s", "--save",
        default=".",
        help="Directory or filename for the .npy adjacency matrix"
    )

    args = parser.parse_args()

    rg = RoboGraph(model_xml_path=args.input, feature_conf_path=args.config)
    rg.build()
    rg.save(save_dir=args.save)

if __name__ == "__main__":
    main()