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
        "-c", "--conf",
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "-s", "--save",
        default=".",
        help="Directory or filename for the .npy adjacency matrix"
    )

    args = parser.parse_args()

    rg = RoboGraph(model_xml_path=args.input, conf_path=args.conf).build()
    rg.save(save_dir=args.save)

if __name__ == "__main__":
    main()