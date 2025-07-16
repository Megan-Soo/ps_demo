import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import pyvista as pv
import re,time

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

def split_frames(nodes, edges, num_frames=60):
    """
    Given an array of nodes and some connections (edges), split the connections into equal sections (by number of frames).

    """
    edges = np.array(edges)
    num_edges = len(edges)

    # Create lists of index ranges
    edges_per_frame = np.linspace(1, num_edges, num_frames, dtype=int)

    frames = []

    for i in range(num_frames):
        # Use each index range to extract connections for each frame
        num_edges_in_frame = edges_per_frame[i]
        current_edges = edges[:num_edges_in_frame]

        # Extract node indices used in current edges
        used_node_indices = np.unique(current_edges)

        # Get the subset of nodes
        current_nodes = nodes[used_node_indices]

        # Map global indices to local indices (for drawing)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_indices)}
        remapped_edges = np.array([[index_map[a], index_map[b]] for a, b in current_edges])

        frames.append({
            'nodes': current_nodes,
            'edges': remapped_edges,
            'original_node_indices': used_node_indices  # if needed
        })

    return frames

# === READ DATA ===

# Read tree
nodes = extract_coordinates('grown.ipnode')
edges = extract_global_numbers('grown.ipelem')
radius = extract_radius('grown_radius.ipfiel')

# Read mesh
mesh = pv.read('peeled.ply')

# === PREP DATA ===

edges = edges - 1 # IMPORTANT: python is zero-indexing, so adjust connections accordingly

# Split tree into frames. each frame has more branches
frames = split_frames(nodes,edges,num_frames=60)

# Make 1st frame the upper airway centreline (14 nodes)
frames[0] = {'nodes':nodes[:15],'edges':edges[:14]}

# Get vertices and faces from mesh
vertices = mesh.points
if mesh.faces.size > 0:
    face_array = mesh.faces.reshape((-1, 4))  # Assuming triangular faces
    faces = face_array[:, 1:]  # shape: (N_faces, 3)
else:
    print(f'No faces?')
    faces = []

# === POLYSCOPING BEGINS ===

ps.init()

# Visualise upper airway to start
upper_air = ps.register_curve_network("upper airway",nodes[:15],edges[:14],radius=0.0007,color=[0.9,1.0,0.1])

# Visualise surface mesh
ps_mesh = ps.register_surface_mesh("mesh",vertices,faces,smooth_shade=True,transparency=0.5,color=[0.9,0.0,0.8])

# == CALLBACK FUNCTION FOR ANIMATION ==

n_frames = len(frames) # number of frames
curr_frame = 0 # parameter to manipulate thru UI
auto_playing = False

def callback():
    global curr_frame, auto_playing

    update_frame_data = False
    _, auto_playing = psim.Checkbox("Autoplay", auto_playing)

    # Advance the frame
    if auto_playing:
        update_frame_data = True
        curr_frame = (curr_frame + 1) % n_frames
        time.sleep(0.05) # add latency to slow down animation (optional)

    # Slider to manually scrub through frames  
    slider_updated, curr_frame = psim.SliderInt("Curr Frame", curr_frame, 0, n_frames-1)
    update_frame_data = update_frame_data or slider_updated

    # Update the scene content if-needed
    if update_frame_data:
        nodes = frames[curr_frame]['nodes']
        edges = frames[curr_frame]['edges']
        tree = ps.register_curve_network("tree",nodes,edges,radius=0.0007,color=[0.9,1.0,0.1])

ps.set_user_callback(callback)

cor_plane_pos = ps.add_scene_slice_plane()
cor_plane_pos.set_pose([0,0,0],[0,1,0])
cor_plane_neg = ps.add_scene_slice_plane()
cor_plane_neg.set_pose([0,0,0],[0,-1,0])
cor_plane_neg.set_active(False)
ps.set_ground_plane_mode("none")
ps.set_navigation_style("free")
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
ps.set_background_color([0,0,0])
ps.show()