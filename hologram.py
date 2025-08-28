import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import SimpleITK as sitk
import pyvista as pv
import os, time, re

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

def get_descendants(nodes, edges, edge_idx, verbose=False):
    """
    Find descendants of the edge at edges[edge_idx].

    Returns:
        descendant_nodes: np.array of shape (M, 3)
        descendant_edges: np.array of shape (K, 2) with indices relative to descendant_nodes
        descendant_node_indices: list of original node indices
        descendant_edge_indices: list of original edge indices
    """
    from collections import defaultdict, deque
    import numpy as np

    num_nodes = nodes.shape[0]
    if verbose:
        print(f"Checking edges: max index = {edges.max()}, node count = {num_nodes}")

    # Validate edges indices
    if edges.max() >= num_nodes or edges.min() < 0:
        raise ValueError(f"Edges contain invalid node indices "
                         f"(must be in [0, {num_nodes-1}]). "
                         f"Found max edge index: {edges.max()}")

    if edge_idx >= len(edges):
        raise IndexError(f"edge_idx {edge_idx} out of bounds for edges with shape {edges.shape}")

    # Build directed graph: parent node → [(child node, edge_idx), ...]
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, i))

    start_node = edges[edge_idx][1]

    visited_nodes = set()
    visited_edges = set()
    queue = deque([start_node])

    while queue:
        current = queue.popleft()
        if current in visited_nodes:
            continue
        visited_nodes.add(current)
        for neighbor, e_idx in graph.get(current, []):
            if e_idx not in visited_edges:
                visited_edges.add(e_idx)
                queue.append(neighbor)

    descendant_node_indices = sorted(visited_nodes)
    descendant_edge_indices = sorted(visited_edges)

    descendant_nodes = nodes[descendant_node_indices]
    descendant_edges_orig = edges[descendant_edge_indices]

    # Reindex edges to descendant_nodes indices (0-based)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(descendant_node_indices)}
    descendant_edges = np.array([
        [old_to_new[u], old_to_new[v]] for u, v in descendant_edges_orig
    ])

    # Final check: all edge indices should be in [0, len(descendant_nodes) - 1]
    max_idx = descendant_edges.max()
    if max_idx >= len(descendant_nodes):
        raise RuntimeError(f"Reindexed edges contain invalid node index {max_idx} (>= {len(descendant_nodes)})")

    return np.array(descendant_nodes), np.array(descendant_edges), np.array(descendant_node_indices), np.array(descendant_edge_indices)

ps.init()

# === SCENE 1: extracting upper airway + lobe geometries from 3D image ===
# Register 3D image volume
img = sitk.ReadImage('ps_demo_data/raw_3D_RAS.mha')
spacing = img.GetSpacing()
arr_img = sitk.GetArrayFromImage(img) # in [Z,Y,X]
shape = arr_img.shape # in (x,y,z)
swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]
img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0),
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=False)
img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',enabled=True)

target = () # set target to centre of block (for turntable revolution)

# Register lobe surface meshes
meshes = []
lobes = ['RUL','RML','RLL','LUL','LLL']
for lobe in lobes:
    meshes.append(pv.read(f'../paper1/results/001/{lobe}.ply'))

# Set up coronal slice planes before setting up meshes (bc meshes getting overwritten in for loop)
# Negative slice plane cuts 3D image, ignores lobes+airway
# Positive slice plane ignores 3D image, cuts lobes+airway

# Iteratively register and set properties of lobe surface meshes
grp_mesh = ps.create_group("meshes")
for (lobe,mesh) in zip(lobes,meshes):
    vertices = mesh.points
    if mesh.faces.size > 0:
        face_array = mesh.faces.reshape((-1, 4))  # Assuming triangular faces
        faces = face_array[:, 1:]  # shape: (N_faces, 3)
    surf = ps.register_surface_mesh(f'{lobe}',vertices,faces, transparency=0.05,smooth_shade=True)
    surf.add_to_group('meshes')
    surf.set_ignore_slice_plane(sag_neg2,True)
    surf.set_ignore_slice_plane(sag_pos2,True)
grp_mesh.set_enabled(True)
grp_mesh.set_show_child_details(False)

# Read in grown airways
nodes = extract_coordinates('results/001_multismooth/grown.ipnode')
edges = extract_global_numbers('results/001_multismooth/grown.ipelem')
radius = extract_radius('results/001_multismooth/grown_radius.ipfiel')
edges = edges-1
radius, _ = compute_joint_radii(nodes, edges, radius)

# Get upper airway nodes and edges
nodes_upper = nodes[:14]
edges_upper = edges[:13]
radius_upper = radius[:14]

upper_nodes = ps.register_curve_network("upper airway",nodes_upper,edges_upper)
upper_nodes.add_scalar_quantity("radius", radius_upper, defined_on='nodes', enabled=False)
upper_nodes.set_node_radius_quantity("radius")

# === SCENE 2: tree growing into lobes ===
# Break down tree into lobes
lobe_sub_nodes = []
lobe_sub_edges = []
lobe_sub_rad = []
parents = [10,12,13,8,9]
for parent in parents:
    sub_n, sub_e, idx_rad, _ = get_descendants(nodes,edges,parent-1)
    lobe_sub_nodes.append(sub_n)
    lobe_sub_edges.append(sub_e)
    lobe_sub_rad.append(radius[idx_rad])

# === END SCENE: 2D image of logo & credits ===
# create end plane

# Register 2D image. Position in front of model
w = 1024
h = 768

ps.add_color_image_quantity("color_img", np.zeros((h, w, 3)), enabled=True, 
                            show_fullscreen=True, show_in_camera_billboard=False, transparency=0.5)

ps.add_color_alpha_image_quantity("color_alpha_img", np.zeros((h, w, 4)), enabled=True, 
                                  show_in_imgui_window=True, show_in_camera_billboard=False,
                                  is_premultiplied=True, image_origin='lower_left')

ps.add_scalar_image_quantity("scalar_img", np.zeros((h, w)), enabled=True, 
                             datatype='symmetric', vminmax=(-3.,.3), cmap='reds')

# set plane to affect 2D image only

# == CALLBACK FUNCTION FOR ANIMATION ==
# Button for camera movement: simple turntable revolution

# SCENE 1 slice plane movement: anterior to posterior
# SCENE 2 tree growing: generation by generation
# SCENE 3 reset view, tissue units ventilate, slice plane cuts into model, revealing defect region 
# SCENE 4: cam tracing main airway path to defect region (simulated bronchoscopy)
# END SCENE: reset to turntable revolution. Visible slice plane slides in front of cam,
#   revealing image: ABI logo; Lung & Respiratory Group 2025; Editor: Megan Soo; Made with Polyscope

n_frames = len(frames) # number of frames
curr_frame = 0 # parameter to manipulate thru UI
spin = False

def callback():
    global curr_frame, spin

    _, spin = psim.Checkbox("Spin", spin)
    
    while spin: # for turntable revolution, set camera's z to a fixed height, and fix up dir
        for f in range(n_spin):  # edit to make it endless
            angle = 2 * np.pi * f / n_spin 
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = elevation
            camera_pos = (x, y, z) # update cam position
            ps.look_at_dir(camera_pos, target, up_dir=(0,0,1), fly_to=False) # set camera to look at the object

    if(psim.Button("SCENE 1")): # SCENE 1 button
        for t in np.linspace(0., 2*np.pi, 120): # Animate the plane sliding along the scene
            pos = np.cos(t) * .8 + .2
            ps_plane.set_pose((0., 0., pos), (0., 0., -1.))
        
        img_block.set_enabled(False) # Disable 3D image at the end, leaving behind lobe meshes + upper airway
        # might need to disable planes too?

    if(psim.Button("SCENE 2")): # SCENE 2 button
        for curr_frame in range(n_frames_grow): # Grow tree
            nodes = frames_tree[curr_frame]['nodes']
            edges = frames_tree[curr_frame]['edges']
            radius = frames_tree[curr_frame]['radius']
            tree = ps.register_curve_network("tree",nodes,edges)
            tree.add_scalar_quantity("radius", radius, defined_on='nodes', enabled=False)
            tree.set_node_radius_quantity("radius")
            time.sleep(0.05)
        
        pts = ps.register_point_cloud("units",pts) # Register acinus tissue units
        pts.add_scalar_quantity("vol", init_vol)
        pts.set_point_radius_quantity("vol")

        grp_mesh.set_enabled(True) # Turn off lobe meshes at the end, leaving behind 3D geometric model

    if(psim.Button("SCENE 3")): # SCENE 3 button
        spin = False # turn off cam spin if it's on
        ps.look_at_dir() # reset to coronal front view
        for curr_frame in range(n_frames_vent):
            vol = frames_vol[curr_frame]
            pts.add_scalar_quantity("vol",vol)
            pts.set_point_radius_quantity("vol")
            time.sleep(0.05)

    # if(psim.Button("SCENE 4")): # SCENE 4 button
    #     spin = False # turn off cam spin if on
    #     # cam fly to start pt of airway path
    #     for t in range(n_frames_path): # move from node to node?
    #         x = 
    #         y = 
    #         z = 
    #         camera_pos = (x, y, z)
    #         ps.look_at_dir(camera_pos, target, up_dir, fly_to=False) # set camera to look at the object

    # if(psim.Button("END SCENE")): # END SCENE button
        spin = False
        ps.look_at_dir() # reset to coronal front view

        plane_end.set_enabled(True)
        for t in np.linspace(0., 2*np.pi, 120): # Animate the plane sliding along the scene
            pos = np.cos(t) * .8 + .2
            plane_end.set_pose((0., 0., pos), (0., 0., -1.))

    # if(psim.Button("FULL MOVIE")): # FULL MOVIE button
    #     # play scenes chronologically

ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.set_background_color([0,0,0])
ps.show()