"""import mujoco
from pathlib import Path
import os

# Get the absolute path to go1.xml
current_dir = Path(__file__).parent
xml_path = current_dir / "go1.xml"

try:
    xml = xml_path.read_text()
    model = mujoco.MjModel.from_xml_string(xml)
    
    print("\n=== First Joint Attributes ===")
    joint0 = model.joint(0)
    print("Available attributes:", [attr for attr in dir(joint0) if not attr.startswith('_')])
    
    # Check specific attributes
    for attr in ['damping', 'armature', 'range', 'type']:
        try:
            print(f"{attr}: {getattr(joint0, attr)}")
        except AttributeError:
            print(f"{attr}: Not available")
    
    print("\n=== First Body Attributes ===")
    body0 = model.body(0)
    print("Available attributes:", [attr for attr in dir(body0) if not attr.startswith('_')])
    
    # Check specific attributes
    for attr in ['mass', 'pos', 'inertia']:
        try:
            print(f"{attr}: {getattr(body0, attr)}")
        except AttributeError:
            print(f"{attr}: Not available")

except FileNotFoundError:
    print(f"Error: Could not find {xml_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir()}")
except Exception as e:
    print(f"Error: {str(e)}")
"""



from robo_graph import RoboGraph

# Initialize with your files
rg = RoboGraph(
    model_xml_path="go1.xml",
    conf_path="feature_conf.yml"
)

# Build the graph and features
rg.build()
print(f"Joint features shape: {rg.joint_features.shape}")
print(f"Body features shape: {rg.body_features.shape}")
# Save the adjacency matrix
if rg.joint_features is not None:
    print(f"Joint features shape: {rg.joint_features.shape}")
    print(f"Sample joint features: {rg.joint_features[0], rg.joint_features[1]}")
else:
    print("No joint features extracted")

if rg.body_features is not None:
    print(f"Body features shape: {rg.body_features.shape}")
    print(f"Sample body features: {rg.body_features[0], rg.body_features[1]}")
else:
    print("No body features extracted")

rg.save("data/output.npy")