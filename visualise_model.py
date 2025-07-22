import polyscope as ps
import numpy as np
import SimpleITK as sitk
import re, os
from collections import defaultdict

def read_exnodedata(file_path,extn='.exnode'):
    """
    Extract data from .exnode
    Return data in dictionary = {(field name, num Components):[array of field values]}
    """
    try:
        file_path, ext = os.path.splitext(file_path)
        if bool(ext) is False:
            ext = extn

        with open((file_path+ext), 'r') as file:
            lines = file.readlines()

        results = {}  # Dictionary to store the results
        for line in lines:
            if ')' in line:
                # Find the closing parenthesis
                close_paren_index = line.find(')')
                # Find the next comma after the closing parenthesis
                next_comma_index = line.find(',', close_paren_index)
                if next_comma_index != -1:
                    # Extract the substring between ')' and ','
                    words_between = line[close_paren_index + 1:next_comma_index].strip()
                else:
                    words_between = line[close_paren_index + 1:].strip()  # If no comma, get till the end

                # Look for "Components=" and check the subsequent character
                components_index = line.find("Components=")
                if components_index != -1:
                    start_index = components_index + len("Components=")
                    if start_index < len(line):  # Ensure there's a character after "Components="
                        subsequent_char = line[start_index]
                        if subsequent_char.isdigit():
                            # Create the key as a tuple (word, integer)
                            key = (words_between, int(subsequent_char))
                            # Add the tuple as a key to the dictionary with a placeholder value
                            results[key] = None  # Placeholder value; replace with desired value

        ## Collect and store index of each node
        indices = [i for i, s in enumerate(lines) if 'Node' in s]
        float_pattern = r"-?\d+\.\d+" # Regex pattern to match float numbers
        int_pattern = r"\b\d+\b" # Regex pattern to match integer numbers
        float_int_pattern = r'[+-]?\d+(?:\.\d+)?' # Regex pattern to match float or integer numbers
        
        list_node_num = []
        for idx in range(len(indices)):
            line = lines[indices[idx]]
            node_number = int(re.findall(int_pattern, line)[-1])
            list_node_num.append(node_number)

        prev_num_components = 0
        iterator = iter(results.items())
        for field in range(len(results)):
            entry = next(iterator)
            key = entry[0]
            components = entry[0][1]
            # print("Key (field name, components):", key) # field name, components
            # print("Components:", components)

            my_list = []
            for idx in range(len(indices)):
                for i in range(components):
                    line = lines[indices[idx]+1+prev_num_components+i]
                    value = float(re.findall(float_int_pattern, line)[0])
                    my_list.append(value)

            if components>1:
                # Reshape into a 2D list
                my_list = [my_list[i:i+components] for i in range(0, len(my_list), components)]

            results[key] = my_list

            prev_num_components = prev_num_components + components

        results[("Node number",None)] = list_node_num
        return results

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

def extract_coordinates(file_path):
    coordinates = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Node number"):
            i += 1
            x = float(lines[i].split(":")[-1].strip())
            i += 1
            y = float(lines[i].split(":")[-1].strip())
            i += 1
            z = float(lines[i].split(":")[-1].strip())
            coordinates.append([x, y, z])
        i += 1
    
    return np.array(coordinates)

def extract_global_numbers(file_path):
    global_numbers = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Element number"):
            while not lines[i].strip().startswith("Enter the 2 global numbers"):
                i += 1
            numbers = list(map(lambda x: int(x), lines[i].split(":")[-1].strip().split()))
            global_numbers.append(numbers)
        i += 1
    
    return np.array(global_numbers)

def extract_radius(file_path):
    radius_values = []
    pattern = re.compile(r'The field variable value is \[ .*?\]: ([\d\.D\+\-]+)')
    
    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                value = match.group(1).replace('D', 'E')  # Convert Fortran-style exponent
                radius_values.append(float(value))
    
    return np.array(radius_values)

def compute_joint_radii(nodes, edges, edge_mid_radii):
    """
    Given a radius for each edge (at midpoint), compute per-node radius
    as the average of connected edge radii. If a node has only one incident
    edge, use that edge's radius directly.
    
    Parameters
    ----------
    nodes : (N,3)
    edges : (E,2)
    edge_mid_radii : (E,)
    
    Returns
    -------
    joint_radii : (N,) per-node radii
    edge_radii : (E,2) per-edge start/end radii
    """
    N = nodes.shape[0]
    E = edges.shape[0]

    # collect radii per node
    incident = defaultdict(list)
    for e, (i0, i1) in enumerate(edges):
        incident[i0].append(edge_mid_radii[e])
        incident[i1].append(edge_mid_radii[e])

    joint_radii = np.zeros(N, dtype=float)
    for i in range(N):
        if len(incident[i]) == 0:
            joint_radii[i] = 0.0
        elif len(incident[i]) == 1:
            # leaf: just use that edge's radius
            joint_radii[i] = incident[i][0]
        else:
            # average of all connected edge radii
            joint_radii[i] = np.mean(incident[i])

    # now expand to per-edge start/end
    edge_radii = np.zeros((E, 2), dtype=float)
    for e, (i0, i1) in enumerate(edges):
        edge_radii[e, 0] = joint_radii[i0]
        edge_radii[e, 1] = joint_radii[i1]

    return joint_radii, edge_radii

"""
In this script, we will visualise the lung model overlaid on the lung image it was derived from.
"""

# === READ DATA ===
img = sitk.ReadImage('ps_demo_data/raw_3D_RAS.mha')
units_dict = read_exnodedata('ps_demo_data/terminal.exnode')
nodes = extract_coordinates('ps_demo_data/grown.ipnode')
edges = extract_global_numbers('ps_demo_data/grown.ipelem')
radius = extract_radius('ps_demo_data/grown_radius.ipfiel')

# === PREP DATA ===
# Prepare image info
spacing = img.GetSpacing()
arr_img = sitk.GetArrayFromImage(img) # in [Z,Y,X]
shape = arr_img.shape # in (x,y,z)
swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]

# Prepare terminal units info
coordinates = np.array(units_dict[('coordinates',3)])
flow = np.array(units_dict[('flow',1)])

# Prepare tree info
edges = edges - 1 # IMPORTANT: python is zero-indexing, so adjust connections accordingly

# joint_radii, _ = compute_joint_radii(nodes, edges, radius)

# === POLYSCOPING BEGINS ===
ps.init()

# Define a 3D volume using the Bounding Box of the 3D image in [X,Y,Z] format
img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0),
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=False)
# Add image intensity values to the 3D volume in [X,Y,Z] format
img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',enabled=True)

# Register acinus tissue units
terminal = ps.register_point_cloud("terminal",coordinates,radius=0.002,enabled=False)
terminal.add_scalar_quantity("flow",flow,cmap='jet',enabled=True)

# Register tree
tree = ps.register_curve_network("tree",nodes,edges,color=[155/255,155/255,155/255])
# tree.add_scalar_quantity("radius node",joint_radii,defined_on='nodes',enabled=True) # this is sufficient to visualise varied edge radii
# tree.set_node_radius_quantity("radius node") # this is sufficient to visualise varied edge radii
# tree.add_scalar_quantity("radius",radius,defined_on='edges',enabled=False) # uncomment this if you want to visualise nodes too
# tree.set_edge_radius_quantity("radius") # uncomment this if you want to visualise nodes too

# Set up planes
cor_plane_pos = ps.add_scene_slice_plane()
cor_plane_pos.set_pose([0,0,0],[0,1,0])
cor_plane_pos.set_draw_widget(True)
cor_plane_pos.set_active(False)

cor_plane_neg = ps.add_scene_slice_plane()
cor_plane_neg.set_pose([0,0,0],[0,-1,0])
cor_plane_neg.set_draw_widget(True)
cor_plane_neg.set_active(False)

ax_plane_pos = ps.add_scene_slice_plane()
ax_plane_pos.set_pose([0,0,0],[0,0,1])
ax_plane_pos.set_draw_widget(True)
ax_plane_pos.set_active(False)

ax_plane_neg = ps.add_scene_slice_plane()
ax_plane_neg.set_pose([0,0,0],[0,0,-1])
ax_plane_neg.set_draw_widget(True)
ax_plane_neg.set_active(False)

# Misc settings
ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0])
ps.show()
