import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import SimpleITK as sitk
import pyvista as pv
from collections import defaultdict
import os, time, re, math

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

def read_terminaldvdt_bin(binpath):
    """
    Binary file stores the volume of each terminal node at each timestep (assuming that nodes have the same no. of timesteps)
    Returns:
    - sorted terminal node numbers
    - terminal volumes sorted by node numbers and stored in timesteps.
    """
    nodes = []
    all_values = []
    with open(binpath, "rb") as f:
        while True:
            raw_node = f.read(4)
            if not raw_node:
                break
            node = np.frombuffer(raw_node, dtype=np.int32)[0]
            nodes.append(node)

            count = np.frombuffer(f.read(4), dtype=np.int32)[0]
            vals = np.frombuffer(f.read(count * 8), dtype=np.float64)
            all_values.append(vals)

    nodes = np.array(nodes, dtype=np.int32)
    all_values = np.array(all_values)  # shape: (num_nodes, num_timesteps)

    # sort by ascending node number
    sort_idx = np.argsort(nodes)
    nodes_sorted = nodes[sort_idx]
    all_values_sorted = all_values[sort_idx, :]  # reorder rows

    # transpose to (num_timesteps, num_nodes)
    values_by_timestep = all_values_sorted.T
    return nodes_sorted, values_by_timestep

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

def split_frames(nodes, edges, radii, num_frames=60, alpha=5.0):
    """
    Given an array of nodes, radii, and connections (edges), split the connections 
    into progressively smaller sections (nonlinear progression).
    
    Parameters
    ----------
    nodes : np.ndarray
        Array of node coordinates, shape (N, d).
    edges : array-like
        List or array of edges, shape (M, 2).
    radii : np.ndarray
        Array of radius values per node, shape (N,).
    num_frames : int
        Number of frames to split into.
    alpha : float
        Progression exponent. 
        <1 = more edges early, >1 = more edges late, =1 is linear.
    
    Returns
    -------
    frames : list of dict
        Each dict contains:
            'nodes' : np.ndarray, subset of nodes in the frame
            'radii' : np.ndarray, radii corresponding to those nodes
            'edges' : np.ndarray, remapped edges
            'original_node_indices' : np.ndarray, indices of nodes in the original array
    """
    edges = np.array(edges)
    num_edges = len(edges)

    # Nonlinear progression of edges per frame
    edges_per_frame = (np.linspace(0, 1, num_frames) ** alpha) * num_edges
    edges_per_frame = np.ceil(edges_per_frame).astype(int)

    frames = []

    for i in range(num_frames):
        num_edges_in_frame = edges_per_frame[i]
        current_edges = edges[:num_edges_in_frame]

        # Extract node indices used in current edges
        used_node_indices = np.unique(current_edges)

        # Get the subset of nodes and radii
        current_nodes = nodes[used_node_indices]
        current_radii = radii[used_node_indices]

        # Map global indices to local indices (for drawing)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_node_indices)}
        remapped_edges = np.array([[index_map[a], index_map[b]] for a, b in current_edges])

        frames.append({
            'nodes': current_nodes,
            'radius': current_radii,
            'edges': remapped_edges,
            'original_node_indices': used_node_indices
        })

    return frames


ps.set_print_prefix("ABI Lungs & Respiratory Group 2025\n")
ps.init()
ps.set_up_dir("z_up")
ps.set_front_dir("neg_y_front")
# ps.set_navigation_style("free")
# reset_cam = ps.get_camera_view_matrix()

# === SCENE 1: extracting upper airway + lobe geometries from 3D image ===
# Register 3D image volume
img = sitk.ReadImage('ps_demo_data/raw_3D_RAS.mha')
spacing = img.GetSpacing()
arr_img = sitk.GetArrayFromImage(img) # in [Z,Y,X]
shape = arr_img.shape # in (x,y,z)
swap_arr = arr_img.transpose(2,1,0) # rearrange [Z,Y,X] to [X,Y,Z]

bound_low = np.array((0, 0, 0))
bound_high = np.array((shape[2]*spacing[0], shape[1]*spacing[1], -shape[0]*spacing[2]))
centre = ((bound_low+bound_high)/2) # set target to centre of block (for turntable revolution)
ps.set_view_center(centre, fly_to=False) # set new centre for turntable navigation

# Register lobe surface meshes
meshes = []
lobes = ['RUL','RML','RLL','LUL','LLL']
for lobe in lobes:
    meshes.append(pv.read(f'../paper1/results/001/{lobe}.ply'))

# Read in grown airways
nodes = extract_coordinates('../lobed_MRI_model/results/001_smooth_once/grown.ipnode')
edges = extract_global_numbers('../lobed_MRI_model/results/001_smooth_once/grown.ipelem')
radius = extract_radius('../lobed_MRI_model/results/001_smooth_once/grown_radius.ipfiel')
edges = edges-1
radius, _ = compute_joint_radii(nodes, edges, radius)

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

lobe_frames = []
n_frames_grow = 60
for i in range(len(lobes)): # Split each lobe tree into frames. each frame has more branches
    lobe_frames.append(split_frames(lobe_sub_nodes[i],lobe_sub_edges[i],lobe_sub_rad[i],num_frames=n_frames_grow)) # returns list of dictionaries

frames = []
for f in range(n_frames_grow):  # for each frame
    f_nodes = []
    f_edges = []
    f_radius = []
    node_offset = 0
    for lobe in lobe_frames:  # for each lobe's frame collection
        n = lobe[f]['nodes']
        e = lobe[f]['edges']
        r = lobe[f]['radii'] if 'radii' in lobe[f] else lobe[f]['radius']
        # Offset edges so node indices are correct in the combined array
        e = e + node_offset
        f_nodes.append(n)
        f_edges.append(e)
        f_radius.append(r)
        node_offset += n.shape[0]
    # Concatenate all lobes for this frame
    f_nodes = np.vstack(f_nodes)
    f_edges = np.vstack(f_edges)
    f_radius = np.concatenate(f_radius)
    frames.append({'nodes': f_nodes, 'edges': f_edges, 'radius': f_radius}) # reorganise so each frame has collection of all lobe trees in that frame

# Make 1st frame the upper airway centreline (14 nodes)
frames[0] = {'nodes':np.array(nodes[:15]),'edges':np.array(edges[:14]),'radius':np.array(radius[:15])}

# Get initial volume
units_dict = read_exnodedata('ps_demo_data/terminal.exnode')

# === SCENE 3: lung ventilating
node_numbers, vals_by_frame = read_terminaldvdt_bin('ps_demo_data/terminaldvdt.bin') # Read terminaldvdt.bin
n_frames_vent = len(vals_by_frame)

# Sort coordinates by Node numbers
combined = list(zip(units_dict[('Node number', None)], units_dict[('coordinates', 3)])) # Zip node numbers with coordinates
combined_sorted = sorted(combined, key=lambda x: x[0]) # Sort both by node number
_, sorted_coords = zip(*combined_sorted) # Unzip the sorted pairs back into separate lists

vols_all_frames = [arr**2+500 for arr in vals_by_frame] # to visibly see unit expansion in animation, scale up values (arbitrary scaling)

# Get min, max values to clamp colormap range. If no clamp, colour no change in animation.
vmin = math.floor(np.min(vols_all_frames))
vmax = math.ceil(np.max(vols_all_frames))


# === END SCENE: 2D image of logo & credits ===
# create end plane

# Register 2D image. Position in front of model
w = 1024
h = 768
ps.add_scalar_image_quantity("scalar_img", np.zeros((h, w)), enabled=False, 
                             datatype='symmetric', vminmax=(-3.,.3), cmap='reds')

# set plane to affect 2D image only


# --- CLEAR MEMORY: delete variables not needed after callback ---
del img, arr_img, bound_high, bound_low
del nodes, edges, radius, lobe_sub_edges, lobe_sub_nodes, lobe_sub_rad, parent, parents, lobe, lobe_frames, f_nodes, f_edges, f_radius, node_offset
del units_dict, combined, combined_sorted, node_numbers, vals_by_frame
# del w, h
import gc
gc.collect()

# == CALLBACK FUNCTION FOR ANIMATION ==
# Button for camera movement: simple turntable revolution

# SCENE 1 slice plane movement: anterior to posterior
# SCENE 2 tree growing: generation by generation
# SCENE 3 reset view, tissue units ventilate, slice plane cuts into model, revealing defect region 
# SCENE 4: cam tracing main airway path to defect region (simulated bronchoscopy)
# END SCENE: reset to turntable revolution. Visible slice plane slides in front of cam,
#   revealing image: ABI logo; Lung & Respiratory Group 2025; Editor: Megan Soo; Made with Polyscope

spin_f = 0 # spin curr frame
n_spin = 500 # spin total frames
spin_radius = shape[1]*spacing[1]
ps.look_at((centre[0],centre[0]-2*spin_radius,centre[2]),centre) # not sure y it doesn't automatically centre cam now. add this to centre cam
spin = False
ext_f = 0 # extraction curr frame
n_extract = 250 # extraction total frames
slice_t = np.linspace(-1, shape[1]*spacing[1], n_extract) # create slice plane positions
extract = False
grow_f = 0
n_grow = n_frames_grow + 10
transparency = np.linspace(0.5,0,n_frames_grow)
grow = False
vent = False
existing = False
# ps.set_view_projection_mode('orthographic') # switch from default 'perspective' projection to orthographic projection
# perspective projection magnifies the change in depth variation as camera rotates, causing 2D slice rotation look 'laggy' as camera position becomes less orthogonal to the slice (facing the slice more diagonally than directly)
# very interesting. Compare it w/ the default 'perspective' projection to see the difference in animation smoothness as camera rotates around the structures.

# # when in orthographic projection, fix intrinsic parameters (esp. field of view) to avoid zooming in/out during animation
# # Suppose your image volume has shape (nx, ny, nz) in world units
# nx, ny, nz = shape
# aspect = nx / ny   # or use viewport aspect ratio

# # Set orthographic "vertical span" big enough for your whole volume
# ortho_span = max(nx, ny, nz) * 1.1   # add margin

# intrinsics = ps.CameraIntrinsics(fov_vertical_deg=ortho_span, aspect=aspect) # fix the intrinsic params to avoid auto updating FOV during animation (which may results in zooming in/out during anime)
# extrinsics = ps.CameraExtrinsics(root=(centre[0],centre[0]-2*spin_radius,centre[2]), look_dir=(0., -1., 0.), up_dir=(0.,1.,0.)) # initial cam position. For revolving camera later, update cam extrinsic params.
# params = ps.CameraParameters(intrinsics, extrinsics)
# ps.set_view_camera_parameters(params)

def callback():
    global spin_f, n_spin, spin_radius, centre, spin
    # vars for Extraction
    global ext_f, n_extract, slice_t, extract, lobes, meshes, swap_arr, shape, spacing
    global meshes_ps, upper_airway
    # vars for growing
    global grow, grow_f, n_grow, n_frames_grow, frames, tree, transparency
    global sorted_coords, vols_all_frames, vmin, vmax, pts
    # vars for ventilation
    global vent, n_frames_vent
    global existing

    update_spin = False
    update_extract = False
    update_grow = False

    _, spin = psim.Checkbox("Spin", spin)
    if spin: # Advance the frame
        if extract:
            ps.set_view_projection_mode('orthographic') # switch from default 'perspective' projection to orthographic projection
        elif grow:
            ps.set_view_projection_mode('perspective')
        update_spin = True
        spin_f = (spin_f + 1) % n_spin
        # time.sleep(0.05) # add latency to slow down animation (optional)
    if update_spin:
        angle = 2 * np.pi * spin_f / n_spin  # full revolution
        # spherical to cartesian camera position        
        x = centre[0] + spin_radius * np.cos(angle)
        y = centre[1] + spin_radius * np.sin(angle)
        z = centre[2] # for turntable revolution, set camera's z to a fixed height
        camera_pos = (x, y, z)
        # set camera to look at the object
        # ps.look_at_dir(camera_pos, centre, (0,0,1), fly_to=False)
        ps.look_at(camera_pos, centre, fly_to=False)

    changed_ext, extract = psim.Checkbox("Extraction", extract)
    if changed_ext:
        if not extract and existing:
            ps.remove_last_scene_slice_plane() # hmm, remove_all_structures doesn't include slice planes
            ps.remove_last_scene_slice_plane()
            ps.remove_all_structures()
            existing = False
            update_extract = False
    if extract: # SCENE 1 button
        update_extract = True
        ext_f = (ext_f + 1) % n_extract
        pos = slice_t[ext_f]

        if not existing: # at the beginning, register all structures
            existing = True
            meshes_ps = []
            for (lobe,mesh) in zip(lobes,meshes):
                vertices = mesh.points
                if mesh.faces.size > 0:
                    face_array = mesh.faces.reshape((-1, 4))  # Assuming triangular faces
                    faces = face_array[:, 1:]  # shape: (N_faces, 3)
                surf = ps.register_surface_mesh(f'{lobe}',vertices,faces,smooth_shade=True, transparency=0.5)
                meshes_ps.append(surf) # store to set ignore planes later

            upper_airway = ps.register_curve_network("upper airway",frames[0]['nodes'],frames[0]['edges'],color=[1.0,0.8,0.8]) # register upper airway
            upper_airway.add_scalar_quantity("radius", frames[0]['radius'], defined_on='nodes')
            upper_airway.set_node_radius_quantity("radius")

            img_block = ps.register_volume_grid("image block", (shape[2],shape[1],shape[0]),bound_low=(0,0,0), # register 3D image
                bound_high=(shape[2]*spacing[0],shape[1]*spacing[1],-shape[0]*spacing[2]),enabled=True)
            img_block.add_scalar_quantity("intensity",swap_arr,defined_on='nodes',enabled=True,cmap='gray')

        if ext_f==0: # at the end, destroy all structures
            ps.remove_all_structures() # free memory
            existing = False
            update_extract=False
    if update_extract: # Animate the plane sliding along the scene
        ps.remove_last_scene_slice_plane() # Remove prev slice plane
        ps.remove_last_scene_slice_plane() # Remove prev slice plane

        # Create updated planes
        cor_plane_neg = ps.add_scene_slice_plane()
        cor_plane_neg.set_pose((0., pos+10, 0.), (0., -1., 0.))
        cor_plane_pos = ps.add_scene_slice_plane()
        cor_plane_pos.set_pose((0., pos, 0.), (0., 1., 0.))

        for m in meshes_ps: # Set ignore meshes
            m.set_ignore_slice_plane(cor_plane_pos,True)

        upper_airway.set_ignore_slice_plane(cor_plane_pos,True)

    changed_grow, grow = psim.Checkbox("Growing", grow)
    if changed_grow:
        if not grow and existing:
            ps.remove_all_structures()
            existing = False
            update_grow = False
    if grow: # SCENE 2 button
        update_grow = True
        grow_f = (grow_f + 1) % n_grow # starts from 1. goes to n_grow-1. then resets to 0.
        time.sleep(0.05)
        
        if not existing:
            ps.set_view_projection_mode('perspective') # not sure if setting it everytime slows it slightly. looks ok for now
            existing = True
            meshes_ps = []
            for (lobe,mesh) in zip(lobes,meshes):
                vertices = mesh.points
                if mesh.faces.size > 0:
                    face_array = mesh.faces.reshape((-1, 4))  # Assuming triangular faces
                    faces = face_array[:, 1:]  # shape: (N_faces, 3)
                surf = ps.register_surface_mesh(f'{lobe}',vertices,faces,smooth_shade=True, transparency=transparency[0],enabled=True)
                meshes_ps.append(surf) # store to set ignore planes later
            
            tree = ps.register_curve_network("upper airway",frames[0]['nodes'],frames[0]['edges'],color=[1.0,0.8,0.8]) # register upper airway
            tree.add_scalar_quantity("radius", frames[0]['radius'], defined_on='nodes')
            tree.set_node_radius_quantity("radius") # nodes become visible in 'orthographic' projection
        
        if grow_f>=n_frames_grow: # once done growing tree, show tissue units
            update_grow = False
            for m in meshes_ps:
                m.set_enabled(False)
            pts = ps.register_point_cloud("units", np.array(sorted_coords),enabled=True) # Register terminal units
            pts.add_scalar_quantity("volume", vols_all_frames[0],vminmax=(vmin,vmax),enabled=True) # varies node colours. Initialise w/ first timestep values.
            pts.set_point_radius_quantity("volume") # varies node size

        if grow_f==0:
            ps.remove_all_structures() # clear memory
            existing = False
            update_grow=False
    if update_grow: # Grow tree
        tree = ps.register_curve_network("airways",frames[grow_f]['nodes'],frames[grow_f]['edges'],color=[1.0,0.8,0.8])
        tree.add_scalar_quantity("radius", frames[grow_f]['radius'], defined_on='nodes', enabled=False)
        tree.set_node_radius_quantity("radius")

        for m in meshes_ps: # meshes fading away
            m.set_transparency(transparency[grow_f])

    # _, vent = psim.Checkbox("Ventilation",vent)
    # while vent: # SCENE 3 button
    #     for curr_frame in range(n_frames_vent):
    #         pts.add_scalar_quantity("volume", vols_all_frames[curr_frame],vminmax=(vmin,vmax),enabled=True)
    #         pts.set_point_radius_quantity("volume")
    #         time.sleep(0.05)

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
        # spin = False
        # ps.set_camera_view_matrix(reset_cam) # reset to coronal front view

        # plane_end.set_enabled(True)
        # for t in np.linspace(0., 2*np.pi, 120): # Animate the plane sliding along the scene
        #     pos = np.cos(t) * .8 + .2
        #     plane_end.set_pose((0., 0., pos), (0., 0., -1.))

    # if(psim.Button("FULL MOVIE")): # FULL MOVIE button
    #     # play scenes chronologically

ps.set_user_callback(callback)
ps.set_ground_plane_mode("none")
ps.set_background_color([0,0,0])
ps.show()